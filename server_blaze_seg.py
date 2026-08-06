# server_blaze_seg.py
# Servidor en tiempo real: captura Basler ToF 101 + segmentación de rocas.
#
# Threads:
#   capture_loop      — captura frames de la cámara (~30fps)
#   segmentation_loop — corre el pipeline ML cada SEG_EVERY frames
#   udp_stream_loop   — envía nube + OBBs de rocas a Blender
#   rpc_server_loop   — RPC TCP: click (x,y,z) → cluster más cercano + prob_roca

import socket
import struct
import threading
import time
import logging
import os
import zlib

import numpy as np
import cv2
import open3d as o3d
from pypylon import pylon

from rock_segmentor import RockSegmentor, SegConfig, SegResult
from simulink_udp import SimulinkUdpSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("server_seg")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UDP_STREAM_IP   = "127.0.0.1"
UDP_STREAM_PORT = 5005
UDP_STREAM_V2_PORT = 5007

TCP_RPC_IP   = "127.0.0.1"
TCP_RPC_PORT = 6006

STRIDE      = 4      # downsample para UDP (sube si va lento)
CONF_MIN    = 0      # filtro de confianza para UDP
SEG_EVERY   = 5      # correr segmentacion cada N frames capturados reales
SEG_STRIDE  = 3      # submuestreo espacial para segmentacion

V2_MAGIC = b"BLZ2"
V2_VERSION = 2
V2_MTU = 60_000
V2_HEADER = struct.Struct("<4sHHQQHHIII")

SEG_CFG = SegConfig(
    conf_min         = 200,
    outlier_nb       = 20,
    outlier_std      = 2.0,
    ransac_dist      = 0.03,
    ransac_n         = 3,
    ransac_iter      = 1000,
    min_height_above = 0.07,
    max_object_pts   = 80_000,
    dbscan_knn       = 15,
    dbscan_eps_pct   = 95.0,
    dbscan_min_pts   = 50,
    prob_threshold   = 0.5,
    model_path       = "models/rock_classifier.joblib",
    scaler_path      = "models/rock_scaler.joblib",
    xyz_scale        = 0.001,
)

# ---------------------------------------------------------------------------
# Estado compartido (lock único)
# ---------------------------------------------------------------------------

LATEST = {
    "xyz":       None,     # (H,W,3) float32
    "depth_bgr": None,     # (H,W,3) uint8
    "conf":      None,     # (H,W) uint16
    "intensity": None,     # (H,W) uint16
    "seg":       None,     # SegResult o None
    "frame_id":  0,        # incrementa una vez por frame capturado
    "seg_frame_id": 0,     # frame_id usado por la ultima segmentacion
    "lock":      threading.Lock(),
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "s"}


def _latest_api_result() -> SegResult | None:
    with LATEST["lock"]:
        return LATEST["seg"]


def _latest_api_status() -> dict:
    with LATEST["lock"]:
        return {
            "source": "camera-server",
            "has_frame": LATEST["xyz"] is not None,
            "frame_id": LATEST["frame_id"],
            "seg_frame_id": LATEST["seg_frame_id"],
        }


def _start_api_thread() -> None:
    from segmentation_api import run_api_server

    host = os.getenv("SEG_API_HOST", "127.0.0.1")
    port = int(os.getenv("SEG_API_PORT", "8000"))
    thread = threading.Thread(
        target=run_api_server,
        kwargs={
            "get_result": _latest_api_result,
            "get_status": _latest_api_status,
            "host": host,
            "port": port,
            "title": "3dcamera Server API",
        },
        daemon=True,
    )
    thread.start()
    log.info("[API] HTTP activa en http://%s:%d", host, port)

# ---------------------------------------------------------------------------
# Helpers de captura (idénticos a server_blaze.py)
# ---------------------------------------------------------------------------

def enable_component(cam, name: str, pixfmt: str) -> None:
    cam.ComponentSelector.Value = name
    cam.ComponentEnable.Value   = True
    try:
        cam.PixelFormat.Value = pixfmt
        log.info("[OK] %s PixelFormat = %s", name, pixfmt)
    except Exception as e:
        log.warning("[WARN] No pude setear %s PixelFormat=%s: %s", name, pixfmt, e)


def reshape_comp(arr, h, w):
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size == h * w:
        return arr.reshape(h, w)
    if arr.ndim == 1 and arr.size == h * w * 3:
        return arr.reshape(h, w, 3)
    return arr


def normalize_to_u8(img, p_lo=2.0, p_hi=98.0):
    x = np.asarray(img, dtype=np.float32)
    mask = np.isfinite(x)
    if mask.sum() < 10:
        return None
    vals = x[mask]
    lo, hi = np.percentile(vals, p_lo), np.percentile(vals, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    y = np.clip(x, lo, hi)
    return np.clip((y - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def build_points(xyz, depth_bgr, conf=None, stride=4, conf_min=0):
    xyz_s = xyz[::stride, ::stride, :]
    col_s = depth_bgr[::stride, ::stride, :]
    pts   = xyz_s.reshape(-1, 3).astype(np.float32)
    cols  = col_s.reshape(-1, 3).astype(np.uint8)

    valid  = np.isfinite(pts).all(axis=1)
    valid &= (np.linalg.norm(pts, axis=1) > 1e-6)
    if conf is not None:
        valid &= (conf[::stride, ::stride].reshape(-1) >= conf_min)

    return pts[valid], cols[valid]


def downsample_frame(xyz, conf, intens, stride):
    if stride <= 1:
        return xyz, conf, intens
    conf_ds = conf[::stride, ::stride] if conf is not None else None
    intens_ds = intens[::stride, ::stride] if intens is not None else None
    return xyz[::stride, ::stride, :], conf_ds, intens_ds


def _fmt_xyz(point: np.ndarray) -> str:
    return f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"

# ---------------------------------------------------------------------------
# capture_loop
# ---------------------------------------------------------------------------

def capture_loop() -> None:
    tl   = pylon.TlFactory.GetInstance()
    devs = tl.EnumerateDevices()
    if not devs:
        raise SystemExit("No se detecta cámara Blaze.")

    cam = pylon.InstantCamera(tl.CreateDevice(devs[0]))
    cam.Open()
    log.info("Cámara: %s | SN: %s",
             cam.GetDeviceInfo().GetModelName(),
             cam.GetDeviceInfo().GetSerialNumber())

    enable_component(cam, "Intensity",  "Mono16")
    enable_component(cam, "Range",      "Coord3D_ABC32f")
    enable_component(cam, "Confidence", "Confidence16")

    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    log.info("[CAP] grabbing...")

    while cam.IsGrabbing():
        grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab.GrabSucceeded():
                continue
            dc = grab.GetDataContainer()

            intensity = xyz = conf = None
            for i in range(dc.DataComponentCount):
                comp = dc.GetDataComponent(i)
                try:
                    arr = reshape_comp(comp.Array, comp.Height, comp.Width)
                    ct  = comp.ComponentType
                    if ct == pylon.ComponentType_Intensity:
                        intensity = arr
                    elif ct == pylon.ComponentType_Range:
                        xyz = arr
                    elif ct == pylon.ComponentType_Confidence:
                        conf = arr
                finally:
                    comp.Release()

            if xyz is None:
                continue

            try:
                # Corregir Z negativo si fuera necesario
                z = xyz[:, :, 2].astype(np.float32)
                finite_z = z[np.isfinite(z)]
                if finite_z.size > 0 and np.nanmedian(finite_z) < 0:
                    xyz = xyz.copy()
                    xyz[:, :, 2] = -z
                    z = xyz[:, :, 2]

                zu8 = normalize_to_u8(z, 2, 98)
                if zu8 is None:
                    continue
                depth_bgr = cv2.applyColorMap(zu8, cv2.COLORMAP_TURBO)
            except Exception as exc:
                log.warning("Frame descartado (error procesando depth): %s", exc)
                continue

            with LATEST["lock"]:
                LATEST["xyz"]       = xyz.astype(np.float32)
                LATEST["depth_bgr"] = depth_bgr
                LATEST["conf"]      = conf
                LATEST["intensity"] = intensity
                LATEST["frame_id"] += 1

        finally:
            grab.Release()

# ---------------------------------------------------------------------------
# segmentation_loop
# ---------------------------------------------------------------------------

def segmentation_loop() -> None:
    segmentor  = RockSegmentor(SEG_CFG)
    simulink_sender = SimulinkUdpSender.from_env()
    last_processed_frame_id = -SEG_EVERY

    while True:
        with LATEST["lock"]:
            xyz  = LATEST["xyz"]
            conf = LATEST["conf"]
            intens = LATEST["intensity"]
            frame_id = LATEST["frame_id"]

        if xyz is None:
            time.sleep(0.01)
            continue

        # Procesa frames reales, sin cola: si se atrasa, toma el ultimo disponible.
        if frame_id == last_processed_frame_id or frame_id - last_processed_frame_id < SEG_EVERY:
            time.sleep(0.005)
            continue

        seg_xyz, seg_conf, seg_intens = downsample_frame(xyz, conf, intens, SEG_STRIDE)
        source_frame_id = frame_id

        result = segmentor.process(seg_xyz, seg_conf, seg_intens)
        last_processed_frame_id = source_frame_id
        if result.clusters:
            main_cluster = result.clusters[0]
            top_xyz = _fmt_xyz(main_cluster.top_face_center)
            simulink_sender.send_xyz(main_cluster.top_face_center, result.tracking_status)
        else:
            top_xyz = "-"

        log.info(
            "[SEG] frame=%d stride=%d | prob_roca=%.3f | is_rock=%s | clusters=%d | track=%s | top_xyz_cam=%s | %.1f ms",
            source_frame_id, SEG_STRIDE,
            result.prob_roca, result.is_rock, len(result.clusters),
            result.tracking_status, top_xyz, result.elapsed_ms
        )

        with LATEST["lock"]:
            LATEST["seg"] = result
            LATEST["seg_frame_id"] = source_frame_id

# ---------------------------------------------------------------------------
# udp_stream_loop — nube coloreada + OBBs de rocas
# ---------------------------------------------------------------------------

def _pack_obb_payload(seg: SegResult) -> bytes:
    """
    Formato: uint16 n_clusters
    Por cluster: float32 prob, uint16 n_corners=8, 8*(float32 x,y,z)
    """
    if seg is None or not seg.clusters:
        return struct.pack("<H", 0)

    buf = struct.pack("<H", len(seg.clusters))
    for c in seg.clusters:
        corners_flat = c.obb_corners.astype(np.float32).tobytes()
        buf += struct.pack("<f", c.prob_roca) + struct.pack("<H", 8) + corners_flat
    return buf


def udp_stream_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        with LATEST["lock"]:
            xyz       = LATEST["xyz"]
            depth_bgr = LATEST["depth_bgr"]
            conf      = LATEST["conf"]
            seg       = LATEST["seg"]

        if xyz is None or depth_bgr is None:
            time.sleep(0.01)
            continue

        pts, cols = build_points(xyz, depth_bgr, conf, stride=STRIDE, conf_min=CONF_MIN)
        n = pts.shape[0]
        if n == 0:
            time.sleep(0.01)
            continue

        header    = struct.pack("<I", n)
        xyz_bytes = pts.astype(np.float32).tobytes(order="C")
        rgb_bytes = cols[:, ::-1].astype(np.uint8).tobytes(order="C")
        obb_bytes = _pack_obb_payload(seg)

        payload = header + xyz_bytes + rgb_bytes + obb_bytes

        MTU = 60_000
        for i in range(0, len(payload), MTU):
            sock.sendto(payload[i:i + MTU], (UDP_STREAM_IP, UDP_STREAM_PORT))

        time.sleep(0.03)


def _pack_v2_frame_payload(pts: np.ndarray, cols: np.ndarray, seg: SegResult) -> bytes:
    header = struct.pack("<I", int(pts.shape[0]))
    xyz_bytes = pts.astype(np.float32).tobytes(order="C")
    rgb_bytes = cols[:, ::-1].astype(np.uint8).tobytes(order="C")
    return header + xyz_bytes + rgb_bytes + _pack_obb_payload(seg)


def _send_v2_frame(sock: socket.socket, frame_id: int, timestamp_ns: int, payload: bytes) -> None:
    max_chunk_payload = V2_MTU - V2_HEADER.size
    chunk_count = max(1, (len(payload) + max_chunk_payload - 1) // max_chunk_payload)

    for chunk_index in range(chunk_count):
        start = chunk_index * max_chunk_payload
        chunk = payload[start:start + max_chunk_payload]
        header = V2_HEADER.pack(
            V2_MAGIC,
            V2_VERSION,
            V2_HEADER.size,
            int(frame_id),
            int(timestamp_ns),
            int(chunk_index),
            int(chunk_count),
            int(len(chunk)),
            int(len(payload)),
            zlib.crc32(chunk) & 0xFFFFFFFF,
        )
        sock.sendto(header + chunk, (UDP_STREAM_IP, UDP_STREAM_V2_PORT))


def udp_stream_v2_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_sent_frame_id = 0
    log.info("[UDPv2] enviando a %s:%d", UDP_STREAM_IP, UDP_STREAM_V2_PORT)

    while True:
        with LATEST["lock"]:
            xyz       = LATEST["xyz"]
            depth_bgr = LATEST["depth_bgr"]
            conf      = LATEST["conf"]
            seg       = LATEST["seg"]
            frame_id  = LATEST["frame_id"]

        if xyz is None or depth_bgr is None or frame_id == last_sent_frame_id:
            time.sleep(0.002)
            continue

        pts, cols = build_points(xyz, depth_bgr, conf, stride=STRIDE, conf_min=CONF_MIN)
        if pts.shape[0] == 0:
            time.sleep(0.002)
            continue

        pts_m = pts.astype(np.float32, copy=False) * float(SEG_CFG.xyz_scale)
        payload = _pack_v2_frame_payload(pts_m, cols, seg)
        _send_v2_frame(sock, frame_id, time.time_ns(), payload)
        last_sent_frame_id = frame_id

# ---------------------------------------------------------------------------
# rpc_server_loop — click (x,y,z) → cluster más cercano
# ---------------------------------------------------------------------------

def rpc_handle_client(conn) -> None:
    """
    Protocolo:
      Cliente → 12 bytes float32 x,y,z
      Servidor:
        uint8 ok (1/0)
        si ok: 8 corners * float32(x,y,z) = 96 bytes
               float32 prob_roca
               uint16 label_len + label utf-8
        si no: uint16 msg_len + msg utf-8
    """
    try:
        data = conn.recv(12)
        if len(data) != 12:
            return
        x, y, z = struct.unpack("<fff", data)
        pick = np.array([x, y, z], dtype=np.float32)

        with LATEST["lock"]:
            seg = LATEST["seg"]
            xyz = LATEST["xyz"]
            depth_bgr = LATEST["depth_bgr"]
            conf_arr  = LATEST["conf"]

        # Si tenemos segmentación con clusters, buscar el más cercano al click
        if seg is not None and seg.clusters:
            best_c    = None
            best_dist = np.inf
            for c in seg.clusters:
                d = float(np.linalg.norm(c.obb_center - pick))
                if d < best_dist:
                    best_dist = d
                    best_c = c

            if best_c is not None:
                label   = f"roca (p={best_c.prob_roca:.2f})"
                label_b = label.encode("utf-8")
                conn.sendall(
                    struct.pack("<B", 1)
                    + best_c.obb_corners.tobytes()
                    + struct.pack("<f", best_c.prob_roca)
                    + struct.pack("<H", len(label_b)) + label_b
                )
                return

        # Fallback: DBSCAN local en tiempo real (comportamiento original)
        if xyz is None or depth_bgr is None:
            _rpc_err(conn, "no frame yet")
            return

        pts, _ = build_points(xyz, depth_bgr, conf_arr, stride=STRIDE, conf_min=CONF_MIN)
        pts = pts.astype(np.float32, copy=False) * float(SEG_CFG.xyz_scale)
        if pts.shape[0] == 0:
            _rpc_err(conn, "empty cloud")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        kdt = o3d.geometry.KDTreeFlann(pcd)
        _, idx, _ = kdt.search_knn_vector_3d(pick.astype(np.float64), 1)
        if not idx:
            _rpc_err(conn, "no nearest point")
            return

        seed  = pts[idx[0]]
        r     = 0.6
        local = pts[np.linalg.norm(pts - seed, axis=1) < r]
        if local.shape[0] < 200:
            _rpc_err(conn, "local region too small")
            return

        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(local.astype(np.float64))
        labels = np.array(local_pcd.cluster_dbscan(eps=0.03, min_points=50, print_progress=False))
        if labels.max() < 0:
            _rpc_err(conn, "no clusters found")
            return

        j   = int(np.argmin(np.linalg.norm(local - seed, axis=1)))
        cid = int(labels[j]) if labels[j] >= 0 else int(np.bincount(labels[labels >= 0]).argmax())

        c_pts = local[labels == cid]
        c_pcd = o3d.geometry.PointCloud()
        c_pcd.points = o3d.utility.Vector3dVector(c_pts.astype(np.float64))
        obb     = c_pcd.get_oriented_bounding_box()
        corners = np.asarray(obb.get_box_points(), dtype=np.float32)

        label_b = b"object"
        conn.sendall(
            struct.pack("<B", 1)
            + corners.tobytes()
            + struct.pack("<f", 0.0)
            + struct.pack("<H", len(label_b)) + label_b
        )

    except Exception as e:
        _rpc_err(conn, f"rpc error: {e}")
    finally:
        conn.close()


def _rpc_err(conn, msg: str) -> None:
    b = msg.encode("utf-8")
    conn.sendall(struct.pack("<B H", 0, len(b)) + b)


def rpc_server_loop() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((TCP_RPC_IP, TCP_RPC_PORT))
    s.listen(5)
    log.info("[RPC] escuchando en %s:%d", TCP_RPC_IP, TCP_RPC_PORT)
    while True:
        conn, addr = s.accept()
        threading.Thread(target=rpc_handle_client, args=(conn,), daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _env_bool("SEG_API_ENABLED", False):
        _start_api_thread()
    threading.Thread(target=segmentation_loop, daemon=True).start()
    threading.Thread(target=udp_stream_loop,   daemon=True).start()
    threading.Thread(target=udp_stream_v2_loop, daemon=True).start()
    threading.Thread(target=rpc_server_loop,   daemon=True).start()
    capture_loop()
