# viewer_seg.py
# Visualizador en tiempo real: captura Basler ToF 101, segmenta rocas y muestra
# la nube coloreada en Open3D con OBBs por cluster.
#
# USO: python viewer_seg.py
#
# NOTA: este script accede directamente a la cámara.
#       NO ejecutar simultáneamente con server_blaze_seg.py.
#
# Threads:
#   main          — Open3D Visualizer (obligatorio en hilo principal en Windows)
#   capture_loop  — graba frames de la cámara a ~30fps
#   seg_loop      — corre segmentación cada SEG_EVERY frames

from __future__ import annotations

import threading
import time
import logging

import numpy as np
import open3d as o3d
from pypylon import pylon

from rock_segmentor import RockSegmentor, SegConfig, SegResult
from simulink_udp import SimulinkUdpSender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("viewer")

# ---------------------------------------------------------------------------
# Paleta de colores para clusters de roca
# ---------------------------------------------------------------------------
PALETTE = np.array([
    [0.20, 0.60, 1.00],   # azul
    [1.00, 0.40, 0.10],   # naranja
    [0.20, 0.90, 0.30],   # verde lima
    [0.90, 0.20, 0.80],   # magenta
    [1.00, 0.90, 0.00],   # amarillo
    [0.50, 0.10, 0.90],   # violeta
    [0.10, 0.90, 0.90],   # cyan
    [1.00, 0.10, 0.20],   # rojo
    [0.60, 0.90, 0.20],   # verde claro
    [0.90, 0.50, 0.10],   # naranja oscuro
], dtype=np.float64)

COLOR_GROUND = np.array([0.30, 0.30, 0.30])
COLOR_NOISE  = np.array([0.10, 0.10, 0.10])
COLOR_OTHER_OBJECT = np.array([0.18, 0.18, 0.18])
COLOR_CYAN_CORE = np.array([0.00, 0.85, 1.00])
COLOR_LABEL = np.array([1.00, 0.86, 0.05])
COLOR_MARKER = np.array([1.00, 0.08, 0.05])

OBB_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

SEG_EVERY = 5   # corre segmentación cada N frames capturados

SEG_CFG = SegConfig(
    conf_min         = 200,
    outlier_nb       = 20,
    outlier_std      = 2.0,
    ransac_dist      = 0.03,   # tolerancia al ajustar el plano suelo
    ransac_n         = 3,
    ransac_iter      = 1000,
    min_height_above = 0.07,   # ignora relieve/grizzly bajo sobre el suelo
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
# Estado compartido
# ---------------------------------------------------------------------------
LATEST: dict = {
    "xyz":  None,       # (H,W,3) float32
    "conf": None,       # (H,W) uint16
    "intens": None,     # (H,W) uint16
    "seg":  None,       # SegResult
    "lock": threading.Lock(),
    "stop": False,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reshape(arr, h, w):
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size == h * w:
        return arr.reshape(h, w)
    if arr.ndim == 1 and arr.size == h * w * 3:
        return arr.reshape(h, w, 3)
    return arr


def _enable(cam, name, pixfmt):
    cam.ComponentSelector.Value = name
    cam.ComponentEnable.Value   = True
    try:
        cam.PixelFormat.Value = pixfmt
        log.info("[OK] %s = %s", name, pixfmt)
    except Exception as e:
        log.warning("[WARN] %s: %s", name, e)


def _flip_y(pts: np.ndarray) -> np.ndarray:
    """Convierte coordenadas de cámara (Y abajo) a Open3D (Y arriba)."""
    out = pts.copy()
    out[:, 1] = -out[:, 1]
    return out


def _build_raw_cloud(xyz_hw3: np.ndarray, scale=0.001) -> np.ndarray:
    """Nube cruda (sin segmentación) para visualizar el frame en vivo."""
    pts = xyz_hw3.reshape(-1, 3).astype(np.float32) * scale
    valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-4)
    return _flip_y(pts[valid])


def _build_seg_cloud(result: SegResult):
    """
    Construye arrays de puntos y colores a partir de un SegResult.
    Retorna (points_Nx3, colors_Nx3) en float64 con Y invertido para Open3D.
    """
    pts_list, col_list = [], []

    # Suelo — gris
    if result.ground_pts is not None and result.ground_pts.shape[0] > 0:
        gp = result.ground_pts
        pts_list.append(gp)
        col_list.append(np.tile(COLOR_GROUND, (gp.shape[0], 1)))

    # Objetos — coloreados por cluster
    if result.object_pts is not None and result.object_pts.shape[0] > 0:
        op = result.object_pts
        labels = result.object_labels

        if labels is not None:
            cols = np.tile(COLOR_OTHER_OBJECT, (op.shape[0], 1))
            for i, c in enumerate(result.clusters):
                mask = labels == c.cluster_id
                cols[mask] = PALETTE[i % len(PALETTE)]
            cols[labels < 0] = COLOR_NOISE
        else:
            p = float(result.prob_roca)
            cols = np.tile(np.array([p, 0.3, 1.0 - p]), (op.shape[0], 1))

        pts_list.append(op)
        col_list.append(cols)

    if not pts_list:
        return np.zeros((0, 3)), np.zeros((0, 3))

    all_pts  = _flip_y(np.vstack(pts_list)).astype(np.float64)
    all_cols = np.vstack(col_list).astype(np.float64)
    return all_pts, all_cols


def _fmt_xyz(point: np.ndarray) -> str:
    return f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"


def _fmt_xyz_label(point: np.ndarray) -> str:
    return f"X={point[0]:.3f}\nY={point[1]:.3f}\nZ={point[2]:.3f}"


_TEXT_SEGMENTS = {
    "0": [((0, 0), (1, 0)), ((1, 0), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 0))],
    "1": [((0.5, 0), (0.5, 2))],
    "2": [((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0))],
    "3": [((0, 2), (1, 2)), ((1, 2), (1, 0)), ((0, 1), (1, 1)), ((0, 0), (1, 0))],
    "4": [((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 2), (1, 0))],
    "5": [((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((1, 0), (0, 0))],
    "6": [((1, 2), (0, 2)), ((0, 2), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1))],
    "7": [((0, 2), (1, 2)), ((1, 2), (0.4, 0))],
    "8": [((0, 0), (1, 0)), ((1, 0), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 0)), ((0, 1), (1, 1))],
    "9": [((1, 0), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((0, 0), (1, 0))],
    "X": [((0, 0), (1, 2)), ((0, 2), (1, 0))],
    "Y": [((0, 2), (0.5, 1)), ((1, 2), (0.5, 1)), ((0.5, 1), (0.5, 0))],
    "Z": [((0, 2), (1, 2)), ((1, 2), (0, 0)), ((0, 0), (1, 0))],
    "C": [((1, 2), (0, 2)), ((0, 2), (0, 0)), ((0, 0), (1, 0))],
    "H": [((0, 0), (0, 2)), ((1, 0), (1, 2)), ((0, 1), (1, 1))],
    "K": [((0, 0), (0, 2)), ((0, 1), (1, 2)), ((0, 1), (1, 0))],
    "L": [((0, 2), (0, 0)), ((0, 0), (1, 0))],
    "O": [((0, 0), (1, 0)), ((1, 0), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 0))],
    "R": [((0, 0), (0, 2)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (1, 0))],
    "W": [((0, 2), (0.2, 0)), ((0.2, 0), (0.5, 0.8)), ((0.5, 0.8), (0.8, 0)), ((0.8, 0), (1, 2))],
    "=": [((0.1, 1.25), (0.9, 1.25)), ((0.1, 0.75), (0.9, 0.75))],
    ":": [((0.5, 1.35), (0.5, 1.45)), ((0.5, 0.55), (0.5, 0.65))],
    "-": [((0.15, 1), (0.85, 1))],
    ".": [((0.45, 0), (0.55, 0))],
}


def _make_text_label(
    text: str,
    origin: np.ndarray,
    scale: float = 0.012,
    color: tuple[float, float, float] = (1.0, 0.95, 0.05),
    mirror_x: bool = False,
) -> o3d.geometry.LineSet:
    points, lines, colors = [], [], []
    advance = 1.45
    line_gap = 2.65
    line_widths = [
        sum(advance if ch != " " else advance for ch in line)
        for line in text.split("\n")
    ]
    cursor_x = 0.0
    cursor_y = 0.0
    line_idx = 0

    for ch in text:
        if ch == "\n":
            cursor_x = 0.0
            cursor_y -= line_gap
            line_idx += 1
            continue
        if ch == " ":
            cursor_x += advance
            continue

        for p0, p1 in _TEXT_SEGMENTS.get(ch, []):
            x0 = cursor_x + p0[0]
            x1 = cursor_x + p1[0]
            if mirror_x:
                width = line_widths[line_idx]
                x0 = width - x0
                x1 = width - x1
            start = origin + np.array([x0 * scale, (cursor_y + p0[1]) * scale, 0.0])
            end = origin + np.array([x1 * scale, (cursor_y + p1[1]) * scale, 0.0])
            lines.append([len(points), len(points) + 1])
            points.extend([start, end])
            colors.append(color)
        cursor_x += advance

    label = o3d.geometry.LineSet()
    label.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    label.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    label.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return label


def _to_display_point(point_cam: np.ndarray) -> np.ndarray:
    out = point_cam.astype(np.float64).copy()
    out[1] = -out[1]
    return out


def _to_display_points(points_cam: np.ndarray) -> np.ndarray:
    out = points_cam.astype(np.float64).copy()
    out[:, 1] = -out[:, 1]
    return out


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    target = direction.astype(np.float64)
    norm = float(np.linalg.norm(target))
    if norm < 1e-9:
        return np.eye(3)
    target /= norm
    source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    s = float(np.linalg.norm(cross))
    if s < 1e-9:
        return np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + k + k @ k * ((1.0 - dot) / (s * s))


def _make_cylinder_between(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: np.ndarray,
    resolution: int = 10,
) -> o3d.geometry.TriangleMesh:
    start = start.astype(np.float64)
    end = end.astype(np.float64)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(start)
    else:
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=length,
            resolution=resolution,
            split=1,
        )
        mesh.rotate(_rotation_from_z(vec), center=np.zeros(3))
        mesh.translate((start + end) * 0.5)
    mesh.paint_uniform_color(color.astype(np.float64))
    mesh.compute_vertex_normals()
    return mesh


def _combine_meshes(meshes: list[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    combined = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        combined += mesh
    combined.compute_vertex_normals()
    return combined


def _make_obb_lines(
    corners: np.ndarray,
    color: np.ndarray = COLOR_CYAN_CORE,
) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(OBB_EDGES, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(color.astype(np.float64), (len(OBB_EDGES), 1))
    )
    return line_set


def _make_arrow(
    start: np.ndarray,
    end: np.ndarray,
    color: np.ndarray,
    radius: float = 0.002,
) -> o3d.geometry.TriangleMesh:
    start = start.astype(np.float64)
    end = end.astype(np.float64)
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return o3d.geometry.TriangleMesh()

    direction = vec / length
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(ref @ direction)) > 0.85:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    side = np.cross(direction, ref)
    side /= max(float(np.linalg.norm(side)), 1e-6)

    head_len = min(0.055, max(0.025, length * 0.18))
    head_w = head_len * 0.45
    shaft_end = end - direction * (head_len * 0.45)
    head_a = end - direction * head_len + side * head_w
    head_b = end - direction * head_len - side * head_w

    return _combine_meshes([
        _make_cylinder_between(start, shaft_end, radius, color, resolution=8),
        _make_cylinder_between(head_a, end, radius * 1.2, color, resolution=8),
        _make_cylinder_between(head_b, end, radius * 1.2, color, resolution=8),
    ])


def _fmt_dimensions_label(extent: np.ndarray) -> str:
    length = float(max(extent[0], extent[1]))
    width = float(min(extent[0], extent[1]))
    height = float(extent[2])
    return f"L={length:.3f}\nW={width:.3f}\nH={height:.3f}"


def _add_scene_geometries(
    vis: o3d.visualization.Visualizer,
    geometries: list,
    active: list,
) -> None:
    for geom in geometries:
        vis.add_geometry(geom, reset_bounding_box=False)
        active.append(geom)

# ---------------------------------------------------------------------------
# capture_loop — hilo de fondo
# ---------------------------------------------------------------------------

def capture_loop(cam) -> None:
    frame_cnt = 0
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    log.info("[CAP] grabbing...")

    while not LATEST["stop"]:
        try:
            grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
        except Exception as e:
            log.warning("[CAP] timeout: %s", e)
            continue

        try:
            if not grab.GrabSucceeded():
                continue
            dc   = grab.GetDataContainer()
            xyz  = intensity = conf = None

            for i in range(dc.DataComponentCount):
                comp = dc.GetDataComponent(i)
                try:
                    arr = _reshape(comp.Array, comp.Height, comp.Width)
                    ct  = comp.ComponentType
                    if ct == pylon.ComponentType_Range:
                        xyz = arr
                    elif ct == pylon.ComponentType_Intensity:
                        intensity = arr
                    elif ct == pylon.ComponentType_Confidence:
                        conf = arr
                finally:
                    comp.Release()

            if xyz is None:
                continue

            # Corregir Z negativo
            try:
                z = xyz[:, :, 2].astype(np.float32)
                fz = z[np.isfinite(z)]
                if fz.size > 0 and float(np.nanmedian(fz)) < 0:
                    xyz = xyz.copy()
                    xyz[:, :, 2] = -z
            except Exception:
                pass

            with LATEST["lock"]:
                LATEST["xyz"]   = xyz.astype(np.float32)
                LATEST["conf"]  = conf
                LATEST["intens"] = intensity

            frame_cnt += 1
        except Exception as e:
            log.warning("[CAP] frame error: %s", e)
        finally:
            grab.Release()

    cam.StopGrabbing()

# ---------------------------------------------------------------------------
# seg_loop — hilo de fondo
# ---------------------------------------------------------------------------

def seg_loop(segmentor: RockSegmentor) -> None:
    frame_seen = 0
    seg_frame  = 0
    simulink_sender = SimulinkUdpSender.from_env()

    while not LATEST["stop"]:
        with LATEST["lock"]:
            xyz   = LATEST["xyz"]
            conf  = LATEST["conf"]
            intens = LATEST["intens"]

        if xyz is None:
            time.sleep(0.01)
            continue

        # Throttle: solo cada SEG_EVERY veces que hay un frame nuevo
        frame_seen += 1
        if frame_seen % SEG_EVERY != 0:
            time.sleep(0.005)
            continue

        result = segmentor.process(xyz, conf, intens)
        if result.clusters:
            main_cluster = result.clusters[0]
            top_xyz = _fmt_xyz(main_cluster.top_face_center)
            simulink_sender.send_xyz(main_cluster.top_face_center, result.tracking_status)
        else:
            top_xyz = "-"

        log.info("[SEG] prob=%.3f | rock=%s | clusters=%d | track=%s | top_xyz_cam=%s | %.0fms",
                 result.prob_roca, result.is_rock, len(result.clusters),
                 result.tracking_status, top_xyz, result.elapsed_ms)

        with LATEST["lock"]:
            LATEST["seg"] = result

# ---------------------------------------------------------------------------
# Main — Open3D visualizer (hilo principal, obligatorio en Windows)
# ---------------------------------------------------------------------------

def _offline_loop(npz_paths: list) -> None:
    """Carga frames .npz en bucle infinito simulando la cámara."""
    import glob, os
    if not npz_paths:
        raise SystemExit("No se encontraron archivos .npz")
    idx = 0
    while not LATEST["stop"]:
        data = np.load(npz_paths[idx % len(npz_paths)])
        with LATEST["lock"]:
            LATEST["xyz"]    = data["xyz"].astype(np.float32)
            LATEST["conf"]   = data.get("confidence")
            LATEST["intens"] = data.get("intensity")
        idx += 1
        time.sleep(0.1)   # ~10fps simulado


def _latest_api_result() -> SegResult | None:
    with LATEST["lock"]:
        return LATEST["seg"]


def _latest_api_status() -> dict:
    with LATEST["lock"]:
        return {
            "source": "offline" if LATEST.get("offline") else "camera",
            "has_frame": LATEST["xyz"] is not None,
        }


def _start_api_thread(host: str, port: int) -> None:
    from segmentation_api import run_api_server

    thread = threading.Thread(
        target=run_api_server,
        kwargs={
            "get_result": _latest_api_result,
            "get_status": _latest_api_status,
            "host": host,
            "port": port,
            "title": "3dcamera Viewer API",
        },
        daemon=True,
    )
    thread.start()
    log.info("API HTTP activa en http://%s:%d", host, port)


def main(
    offline: str | None = None,
    api: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8000,
) -> None:
    cam = None
    LATEST["offline"] = bool(offline)

    if api:
        _start_api_thread(api_host, api_port)

    if offline:
        import glob
        paths = sorted(glob.glob(offline))
        if not paths:
            raise SystemExit(f"No se encontraron archivos: {offline}")
        log.info("Modo offline — %d frames: %s …", len(paths), paths[0])
        segmentor = RockSegmentor(SEG_CFG)
        threading.Thread(target=_offline_loop, args=(paths,), daemon=True).start()
        threading.Thread(target=seg_loop, args=(segmentor,), daemon=True).start()
    else:
        # --- Abrir cámara ---
        tl   = pylon.TlFactory.GetInstance()
        devs = tl.EnumerateDevices()
        if not devs:
            raise SystemExit("No se detecta cámara Blaze.")
        cam = pylon.InstantCamera(tl.CreateDevice(devs[0]))
        cam.Open()
        log.info("Cámara: %s | SN: %s",
                 cam.GetDeviceInfo().GetModelName(),
                 cam.GetDeviceInfo().GetSerialNumber())
        _enable(cam, "Intensity",  "Mono16")
        _enable(cam, "Range",      "Coord3D_ABC32f")
        _enable(cam, "Confidence", "Confidence16")
        segmentor = RockSegmentor(SEG_CFG)
        threading.Thread(target=capture_loop, args=(cam,), daemon=True).start()
        threading.Thread(target=seg_loop, args=(segmentor,), daemon=True).start()

    # Esperar primer frame
    log.info("Esperando primer frame...")
    while True:
        with LATEST["lock"]:
            if LATEST["xyz"] is not None:
                break
        time.sleep(0.05)

    # --- Open3D Visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Rock Segmentor — Basler ToF 101", width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size        = 2.0
    opt.background_color  = np.array([0.05, 0.05, 0.05])
    opt.show_coordinate_frame = True

    # Nube principal
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # OBBs activos en el visualizador
    active_obbs: list = []
    active_annotations: list = []

    first_reset  = True
    last_seg_ts  = 0.0
    # Cache de última nube segmentada — evita el flash entre segmentaciones
    cached_pts: np.ndarray | None = None
    cached_cols: np.ndarray | None = None

    try:
        while True:
            with LATEST["lock"]:
                xyz = LATEST["xyz"]
                seg = LATEST["seg"]

            if xyz is None:
                time.sleep(0.01)
                if not vis.poll_events():
                    break
                vis.update_renderer()
                continue

            seg_changed = seg is not None and seg.timestamp > last_seg_ts

            # --- Actualizar nube de puntos ---
            if seg_changed:
                # Nueva segmentación: calcular y guardar en caché
                cached_pts, cached_cols = _build_seg_cloud(seg)
                last_seg_ts = seg.timestamp

            if cached_pts is not None:
                # Mostrar última nube segmentada (sin flash)
                pts, cols = cached_pts, cached_cols
            else:
                # Antes de la primera segmentación: nube cruda azul
                pts  = _build_raw_cloud(xyz, scale=SEG_CFG.xyz_scale)
                cols = np.tile([0.4, 0.6, 0.8], (pts.shape[0], 1))

            if pts.shape[0] > 0:
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)
                vis.update_geometry(pcd)
                if first_reset:
                    vis.reset_view_point(True)
                    first_reset = False

            # --- Actualizar OBBs solo cuando hay segmentación nueva ---
            if seg_changed:
                for obb in active_obbs:
                    vis.remove_geometry(obb, reset_bounding_box=False)
                active_obbs.clear()
                for marker in active_annotations:
                    vis.remove_geometry(marker, reset_bounding_box=False)
                active_annotations.clear()

                for i, c in enumerate(seg.clusters):
                    try:
                        corners = _to_display_points(c.obb_corners)
                        box_min = corners.min(axis=0)
                        box_max = corners.max(axis=0)
                        top_point = c.top_face_center_view.astype(np.float64)
                        max_extent = max(float(np.max(c.obb_extent)), 1e-6)
                        label_scale = float(np.clip(max_extent * 0.014, 0.007, 0.013))

                        obb_lines = _make_obb_lines(corners)
                        _add_scene_geometries(vis, [obb_lines], active_obbs)

                        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
                        marker.compute_vertex_normals()
                        marker.paint_uniform_color(COLOR_MARKER.astype(np.float64))
                        marker.translate(top_point)

                        rock_label_origin = top_point + np.array([0.18, 0.16, 0.05])
                        rock_label = _make_text_label(
                            "ROCK",
                            rock_label_origin,
                            scale=label_scale * 0.95,
                            color=tuple(COLOR_LABEL),
                            mirror_x=True,
                        )
                        arrow = _make_arrow(
                            rock_label_origin + np.array([-0.015, -0.020, 0.0]),
                            top_point,
                            COLOR_CYAN_CORE,
                            radius=0.0018,
                        )

                        xyz_origin = np.array([
                            box_min[0] - max_extent * 0.30,
                            box_max[1] + max_extent * 0.08,
                            top_point[2],
                        ])
                        xyz_label = _make_text_label(
                            _fmt_xyz_label(c.top_face_center),
                            xyz_origin,
                            scale=label_scale,
                            color=tuple(COLOR_LABEL),
                            mirror_x=True,
                        )

                        dim_origin = np.array([
                            box_max[0] + max_extent * 0.08,
                            box_max[1] - max_extent * 0.02,
                            top_point[2],
                        ])
                        dim_label = _make_text_label(
                            _fmt_dimensions_label(c.obb_extent),
                            dim_origin,
                            scale=label_scale,
                            color=tuple(COLOR_LABEL),
                            mirror_x=True,
                        )

                        _add_scene_geometries(
                            vis,
                            [marker, arrow, rock_label, xyz_label, dim_label],
                            active_annotations,
                        )
                    except Exception as e:
                        log.warning("OBB %d error: %s", i, e)

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        LATEST["stop"] = True
        if cam is not None:
            cam.Close()
        vis.destroy_window()
        log.info("Visor cerrado.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--offline", default=None,
                   help="Glob de archivos .npz para modo sin cámara. Ej: frames/*.npz")
    p.add_argument("--api", action="store_true",
                   help="Expone la segmentacion actual por HTTP/WebSocket.")
    p.add_argument("--api-host", default="127.0.0.1",
                   help="Host HTTP. Usa 0.0.0.0 para red local o tunel.")
    p.add_argument("--api-port", type=int, default=8000,
                   help="Puerto HTTP para la API.")
    args = p.parse_args()
    main(
        offline=args.offline,
        api=args.api,
        api_host=args.api_host,
        api_port=args.api_port,
    )
