# -*- coding: utf-8 -*-
from pypylon import pylon
import numpy as np
import cv2
import time
import open3d as o3d


def enable_component(cam, name: str, pixfmt: str):
    cam.ComponentSelector.Value = name
    cam.ComponentEnable.Value = True
    try:
        cam.PixelFormat.Value = pixfmt
        print(f"[OK] {name} PixelFormat = {pixfmt}")
    except Exception as e:
        print(f"[WARN] No pude setear {name} PixelFormat={pixfmt}: {e}")


def reshape_comp(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size == h * w:
        return arr.reshape(h, w)
    if arr.ndim == 1 and arr.size == h * w * 3:
        return arr.reshape(h, w, 3)
    return arr


def normalize_to_u8(img: np.ndarray, p_lo=2.0, p_hi=98.0) -> np.ndarray:
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


def build_pointcloud(xyz: np.ndarray, color_bgr: np.ndarray, conf: np.ndarray = None,
                     stride: int = 4, conf_min: int = 0):
    """
    xyz: (H,W,3) float
    color_bgr: (H,W,3) uint8
    conf: (H,W) uint16 opcional
    """
    xyz_s = xyz[::stride, ::stride, :]
    col_s = color_bgr[::stride, ::stride, :]
    conf_s = conf[::stride, ::stride] if conf is not None else None

    pts = xyz_s.reshape(-1, 3).astype(np.float32)
    cols = (col_s.reshape(-1, 3)[:, ::-1] / 255.0).astype(np.float32)  # BGR->RGB

    valid = np.isfinite(pts).all(axis=1)

    # blaze: inválidos típicos (0,0,0)
    valid &= (np.linalg.norm(pts, axis=1) > 1e-6)

    if conf_s is not None:
        c = conf_s.reshape(-1)
        valid &= (c >= conf_min)

    return pts[valid], cols[valid]


def main():
    tl = pylon.TlFactory.GetInstance()
    devs = tl.EnumerateDevices()

    print("Dispositivos detectados:", len(devs))
    for i, d in enumerate(devs):
        print(f"  [{i}] {d.GetFriendlyName()} | {d.GetModelName()} | S/N {d.GetSerialNumber()}")

    if not devs:
        raise SystemExit("No se detecta ninguna cámara.")

    cam = pylon.InstantCamera(tl.CreateDevice(devs[0]))
    cam.Open()
    print("\nCamara:", cam.GetDeviceInfo().GetModelName(), "| SN:", cam.GetDeviceInfo().GetSerialNumber())

    enable_component(cam, "Intensity", "Mono16")
    enable_component(cam, "Range", "Coord3D_ABC32f")      # X,Y,Z float
    enable_component(cam, "Confidence", "Confidence16")

    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("Grab iniciado. Presiona 'q' o ESC para salir.")

    # Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud (blaze)", width=1100, height=800, visible=True)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    fitted = False
    last_print = 0.0

    while cam.IsGrabbing():
        grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab.GrabSucceeded():
                continue

            dc = grab.GetDataContainer()

            intensity = None
            xyz = None
            confidence = None

            for i in range(dc.DataComponentCount):
                comp = dc.GetDataComponent(i)
                try:
                    arr = reshape_comp(comp.Array, comp.Height, comp.Width)

                    if comp.ComponentType == pylon.ComponentType_Intensity:
                        intensity = arr
                    elif comp.ComponentType == pylon.ComponentType_Range:
                        xyz = arr
                    elif comp.ComponentType == pylon.ComponentType_Confidence:
                        confidence = arr
                finally:
                    comp.Release()

            # Mostrar intensity
            if intensity is not None:
                iu8 = normalize_to_u8(intensity, 1, 99)
                if iu8 is not None:
                    cv2.imshow("Intensity (gray)", iu8)

            depth_bgr = None
            if xyz is not None and xyz.ndim == 3 and xyz.shape[2] >= 3:
                z = xyz[:, :, 2].astype(np.float32)

                # Si Z viene mayoritariamente negativa, invierte solo para visualizar
                zmed = np.nanmedian(z[np.isfinite(z)])
                if np.isfinite(zmed) and zmed < 0:
                    z = -z
                    xyz = xyz.copy()
                    xyz[:, :, 2] = z

                zu8 = normalize_to_u8(z, 2, 98)
                if zu8 is not None:
                    depth_bgr = cv2.applyColorMap(zu8, cv2.COLORMAP_TURBO)
                    cv2.imshow("Depth/Z (TURBO)", depth_bgr)

            # Mostrar confidence
            if confidence is not None:
                cu8 = normalize_to_u8(confidence, 2, 98)
                if cu8 is not None:
                    cv2.imshow("Confidence (VIRIDIS)", cv2.applyColorMap(cu8, cv2.COLORMAP_VIRIDIS))

            # Nube de puntos (color por depth)
            if (xyz is not None) and (depth_bgr is not None):
                pts, cols = build_pointcloud(
                    xyz=xyz,
                    color_bgr=depth_bgr,
                    conf=confidence,
                    stride=4,     # sube a 6/8 si va lento
                    conf_min=0
                )

                now = time.time()
                if now - last_print > 1.0:
                    zshow = xyz[:, :, 2]
                    fin = np.isfinite(zshow)
                    if fin.any():
                        print(f"[pc] points={pts.shape[0]} | Z(min/med/max)={np.nanmin(zshow):.3f}/{np.nanmedian(zshow[fin]):.3f}/{np.nanmax(zshow):.3f}")
                    else:
                        print(f"[pc] points={pts.shape[0]} | Z = no finite")
                    last_print = now

                if pts.shape[0] > 0:
                    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

                    vis.update_geometry(pcd)

                    # Si la ventana se cerró, salimos
                    if not vis.poll_events():
                        break
                    vis.update_renderer()

                    # “Fit” compatible (sin fit_bounds)
                    if not fitted:
                        vis.reset_view_point(True)
                        fitted = True

            # Teclas (OpenCV)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        finally:
            grab.Release()

    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("Cerrado OK.")


if __name__ == "__main__":
    main()
