# -*- coding: utf-8 -*-
# server_blaze.py
# Stream UDP: manda point cloud (xyzrgb) a Blender
# RPC TCP: recibe click (x,y,z) y devuelve bbox OBB (8 corners) + label

import socket
import struct
import threading
import time

import numpy as np
import cv2
import open3d as o3d
from pypylon import pylon


UDP_STREAM_IP = "127.0.0.1"
UDP_STREAM_PORT = 5005

TCP_RPC_IP = "127.0.0.1"
TCP_RPC_PORT = 6006

STRIDE = 4          # downsample (sube si va lento)
CONF_MIN = 0        # sube (p.ej. 200) si quieres filtrar ruido
CLUSTER_EPS = 0.03  # metros aprox (ajusta)
CLUSTER_MINPTS = 50

# Estado compartido (última nube)
LATEST = {
    "xyz": None,          # (H,W,3) float32
    "depth_bgr": None,    # (H,W,3) uint8
    "conf": None,         # (H,W) uint16
    "lock": threading.Lock()
}


def enable_component(cam, name: str, pixfmt: str):
    cam.ComponentSelector.Value = name
    cam.ComponentEnable.Value = True
    try:
        cam.PixelFormat.Value = pixfmt
        print(f"[OK] {name} PixelFormat = {pixfmt}")
    except Exception as e:
        print(f"[WARN] No pude setear {name} PixelFormat={pixfmt}: {e}")


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
    lo = np.percentile(vals, p_lo)
    hi = np.percentile(vals, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    y = np.clip(x, lo, hi)
    y = (y - lo) * (255.0 / (hi - lo))
    return np.clip(y, 0, 255).astype(np.uint8)


def build_points(xyz, depth_bgr, conf=None, stride=4, conf_min=0):
    xyz_s = xyz[::stride, ::stride, :]
    col_s = depth_bgr[::stride, ::stride, :]
    conf_s = conf[::stride, ::stride] if conf is not None else None

    pts = xyz_s.reshape(-1, 3).astype(np.float32)
    cols = col_s.reshape(-1, 3).astype(np.uint8)

    valid = np.isfinite(pts).all(axis=1)
    valid &= (np.linalg.norm(pts, axis=1) > 1e-6)
    if conf_s is not None:
        c = conf_s.reshape(-1)
        valid &= (c >= conf_min)

    return pts[valid], cols[valid]


def udp_stream_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        with LATEST["lock"]:
            xyz = LATEST["xyz"]
            depth_bgr = LATEST["depth_bgr"]
            conf = LATEST["conf"]

        if xyz is None or depth_bgr is None:
            time.sleep(0.01)
            continue

        pts, cols = build_points(xyz, depth_bgr, conf, stride=STRIDE, conf_min=CONF_MIN)
        n = pts.shape[0]
        if n == 0:
            time.sleep(0.01)
            continue

        # Empaquetado:
        # uint32 n
        # float32 xyz * n
        # uint8  rgb * n  (en orden RGB)
        header = struct.pack("<I", n)
        xyz_bytes = pts.astype(np.float32).tobytes(order="C")
        rgb = cols[:, ::-1]  # BGR->RGB
        rgb_bytes = rgb.astype(np.uint8).tobytes(order="C")
        payload = header + xyz_bytes + rgb_bytes

        # UDP tiene límite práctico; mandamos en chunks
        MTU = 60000
        for i in range(0, len(payload), MTU):
            sock.sendto(payload[i:i+MTU], (UDP_STREAM_IP, UDP_STREAM_PORT))

        time.sleep(0.03)  # ~33 fps


def rpc_handle_client(conn):
    """
    Protocolo:
    Cliente manda 12 bytes: float32 x,y,z (punto clickeado)
    Servidor responde:
      uint8 ok (1/0)
      si ok:
        8 corners * (float32 x,y,z) = 8*12=96 bytes
        uint16 label_len + label utf-8
      si no ok:
        uint16 msg_len + msg utf-8
    """
    try:
        data = conn.recv(12)
        if len(data) != 12:
            return
        x, y, z = struct.unpack("<fff", data)
        pick = np.array([x, y, z], dtype=np.float32)

        with LATEST["lock"]:
            xyz = LATEST["xyz"]
            depth_bgr = LATEST["depth_bgr"]
            conf = LATEST["conf"]

        if xyz is None or depth_bgr is None:
            msg = "no frame yet".encode("utf-8")
            conn.sendall(struct.pack("<B H", 0, len(msg)) + msg)
            return

        pts, _ = build_points(xyz, depth_bgr, conf, stride=STRIDE, conf_min=CONF_MIN)
        if pts.shape[0] == 0:
            msg = "empty point cloud".encode("utf-8")
            conn.sendall(struct.pack("<B H", 0, len(msg)) + msg)
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        # Busca punto más cercano al pick (para anclar el cluster)
        kdt = o3d.geometry.KDTreeFlann(pcd)
        _, idx, _ = kdt.search_knn_vector_3d(pick.astype(np.float64), 1)
        if not idx:
            msg = "no nearest".encode("utf-8")
            conn.sendall(struct.pack("<B H", 0, len(msg)) + msg)
            return
        seed = pts[idx[0]]

        # Recorte local (esfera) para cluster más estable y rápido
        r = 0.6  # metros (ajusta)
        dist = np.linalg.norm(pts - seed, axis=1)
        local = pts[dist < r]
        if local.shape[0] < 200:
            msg = "local too small".encode("utf-8")
            conn.sendall(struct.pack("<B H", 0, len(msg)) + msg)
            return

        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(local.astype(np.float64))

        labels = np.array(local_pcd.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MINPTS, print_progress=False))
        if labels.max() < 0:
            msg = "no clusters".encode("utf-8")
            conn.sendall(struct.pack("<B H", 0, len(msg)) + msg)
            return

        # elige el cluster que contiene el punto más cercano al seed
        # (aprox: cluster del punto local más cercano a seed)
        dlocal = np.linalg.norm(local - seed, axis=1)
        j = int(np.argmin(dlocal))
        cid = int(labels[j])
        if cid < 0:
            # fallback: mayor cluster
            cid = int(np.bincount(labels[labels >= 0]).argmax())

        cluster_pts = local[labels == cid]
        if cluster_pts.shape[0] < 50:
            msg = "cluster too small".encode("utf-8")
            conn.sendall(struct.pack("<B H", 0, len(msg)) + msg)
            return

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts.astype(np.float64))

        obb = cluster_pcd.get_oriented_bounding_box()
        corners = np.asarray(obb.get_box_points(), dtype=np.float32)  # (8,3)

        # TODO: aquí enchufas YOLO/CLIP
        label = "object"  # placeholder
        label_b = label.encode("utf-8")

        conn.sendall(struct.pack("<B", 1) + corners.tobytes() + struct.pack("<H", len(label_b)) + label_b)

    except Exception as e:
        msg = f"rpc error: {e}".encode("utf-8")
        conn.sendall(struct.pack("<B H", 0, len(msg)) + msg)
    finally:
        conn.close()


def rpc_server_loop():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_RPC_IP, TCP_RPC_PORT))
    s.listen(5)
    print(f"[RPC] listening on {TCP_RPC_IP}:{TCP_RPC_PORT}")
    while True:
        conn, _ = s.accept()
        threading.Thread(target=rpc_handle_client, args=(conn,), daemon=True).start()


def capture_loop():
    tl = pylon.TlFactory.GetInstance()
    devs = tl.EnumerateDevices()
    if not devs:
        raise SystemExit("No se detecta blaze.")
    cam = pylon.InstantCamera(tl.CreateDevice(devs[0]))
    cam.Open()
    print("Camara:", cam.GetDeviceInfo().GetModelName(), "| SN:", cam.GetDeviceInfo().GetSerialNumber())

    enable_component(cam, "Intensity", "Mono16")
    enable_component(cam, "Range", "Coord3D_ABC32f")
    enable_component(cam, "Confidence", "Confidence16")

    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("[CAP] grabbing...")

    while cam.IsGrabbing():
        grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab.GrabSucceeded():
                continue
            dc = grab.GetDataContainer()

            intensity = None
            xyz = None
            conf = None

            for i in range(dc.DataComponentCount):
                comp = dc.GetDataComponent(i)
                try:
                    arr = reshape_comp(comp.Array, comp.Height, comp.Width)
                    if comp.ComponentType == pylon.ComponentType_Intensity:
                        intensity = arr
                    elif comp.ComponentType == pylon.ComponentType_Range:
                        xyz = arr
                    elif comp.ComponentType == pylon.ComponentType_Confidence:
                        conf = arr
                finally:
                    comp.Release()

            if xyz is None:
                continue

            # Depth colormap (para pintar nube y para debug)
            z = xyz[:, :, 2].astype(np.float32)
            zmed = np.nanmedian(z[np.isfinite(z)])
            if np.isfinite(zmed) and zmed < 0:
                xyz = xyz.copy()
                xyz[:, :, 2] = -z
                z = xyz[:, :, 2]

            zu8 = normalize_to_u8(z, 2, 98)
            if zu8 is None:
                continue
            depth_bgr = cv2.applyColorMap(zu8, cv2.COLORMAP_TURBO)

            with LATEST["lock"]:
                LATEST["xyz"] = xyz.astype(np.float32)
                LATEST["depth_bgr"] = depth_bgr
                LATEST["conf"] = conf

        finally:
            grab.Release()


if __name__ == "__main__":
    threading.Thread(target=udp_stream_loop, daemon=True).start()
    threading.Thread(target=rpc_server_loop, daemon=True).start()
    capture_loop()

