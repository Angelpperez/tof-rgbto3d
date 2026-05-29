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

SEG_EVERY = 5   # corre segmentación cada N frames capturados

SEG_CFG = SegConfig(
    conf_min         = 200,
    outlier_nb       = 20,
    outlier_std      = 2.0,
    ransac_dist      = 0.03,   # tolerancia al ajustar el plano suelo
    ransac_n         = 3,
    ransac_iter      = 1000,
    min_height_above = 0.02,   # mínimo 2cm sobre el suelo para ser "objeto"
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
            cols = np.zeros((op.shape[0], 3), dtype=np.float64)
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
        log.info("[SEG] prob=%.3f | rock=%s | clusters=%d | %.0fms",
                 result.prob_roca, result.is_rock, len(result.clusters), result.elapsed_ms)

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


def main(offline: str | None = None) -> None:
    cam = None

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

                for i, c in enumerate(seg.clusters):
                    try:
                        center = c.obb_center.copy().astype(np.float64)
                        center[1] = -center[1]   # flip Y igual que la nube
                        obb = o3d.geometry.OrientedBoundingBox(
                            center=center,
                            R=c.obb_R.astype(np.float64),
                            extent=c.obb_extent.astype(np.float64),
                        )
                        obb.color = PALETTE[i % len(PALETTE)]
                        vis.add_geometry(obb, reset_bounding_box=False)
                        active_obbs.append(obb)
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
    args = p.parse_args()
    main(offline=args.offline)
