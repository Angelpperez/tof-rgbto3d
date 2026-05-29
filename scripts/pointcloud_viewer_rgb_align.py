import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

from pointcloud_viewer_plotly import PointCloudViewerPlotly


def resolve_rgb_path(path_in: Path) -> Path:
    if path_in.is_file():
        return path_in
    if path_in.suffix:
        raise FileNotFoundError(f"No existe RGB: {path_in}")

    exts = [".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"]
    for ext in exts:
        cand = path_in.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No encontre RGB con extensiones conocidas: {path_in}")


def load_npz_xyz_conf(npz_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(str(npz_path))
    if "xyz" not in data.files:
        raise ValueError(f"NPZ sin key 'xyz': {npz_path}")
    xyz = data["xyz"]
    conf = data["confidence"] if "confidence" in data.files else None
    return xyz, conf


def center_crop_to_aspect(img: np.ndarray, target_aspect: float) -> np.ndarray:
    h, w = img.shape[:2]
    cur = w / float(h)
    if abs(cur - target_aspect) < 1e-4:
        return img
    if cur > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = max(0, (w - new_w) // 2)
        return img[:, x0:x0 + new_w]
    new_h = int(round(w / target_aspect))
    y0 = max(0, (h - new_h) // 2)
    return img[y0:y0 + new_h, :]


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return img
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    x0_src = max(0, -dx)
    y0_src = max(0, -dy)
    x1_src = min(w, w - dx)
    y1_src = min(h, h - dy)

    x0_dst = max(0, dx)
    y0_dst = max(0, dy)
    x1_dst = min(w, w + dx)
    y1_dst = min(h, h + dy)

    if x1_src > x0_src and y1_src > y0_src:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out


def align_rgb_to_tof(rgb_bgr: np.ndarray, tof_shape: Tuple[int, int], shift_xy: Tuple[int, int]) -> np.ndarray:
    th, tw = tof_shape
    target_aspect = tw / float(th)
    cropped = center_crop_to_aspect(rgb_bgr, target_aspect)
    aligned = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)
    dx, dy = shift_xy
    return shift_image(aligned, dx=dx, dy=dy)


def build_points_with_colors(
    xyz: np.ndarray,
    rgb_aligned: np.ndarray,
    confidence: Optional[np.ndarray],
    conf_min: Optional[int],
    z_min: float,
    stride: int,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xyz_s = xyz[::stride, ::stride, :]
    rgb_s = rgb_aligned[::stride, ::stride, :]
    conf_s = confidence[::stride, ::stride] if confidence is not None else None

    pts = xyz_s.reshape(-1, 3).astype(np.float32)
    cols = rgb_s.reshape(-1, 3).astype(np.uint8)

    mask = np.isfinite(pts).all(axis=1)
    mask &= (pts[:, 2] > z_min)
    if conf_s is not None and conf_min is not None:
        mask &= (conf_s.reshape(-1) >= conf_min)

    pts = pts[mask]
    cols = cols[mask]

    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    return pts, cols


def apply_axis_flips(
    pts: np.ndarray,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    auto_flip_z: bool,
) -> np.ndarray:
    out = pts.copy()
    if auto_flip_z:
        zmed = np.nanmedian(out[:, 2])
        if np.isfinite(zmed) and zmed < 0:
            out[:, 2] *= -1.0
    if flip_x:
        out[:, 0] *= -1.0
    if flip_y:
        out[:, 1] *= -1.0
    if flip_z:
        out[:, 2] *= -1.0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visor Plotly con RGB alineado (crop 4:3 + resize).")
    p.add_argument("--npz", type=Path, required=True, help="Ruta a NPZ con 'xyz'.")
    p.add_argument("--rgb", type=Path, required=True, help="Ruta a imagen RGB (PNG/JPG/TIFF).")
    p.add_argument("--rgb-shift-x", type=int, default=0, help="Shift X (pix) post-alineacion.")
    p.add_argument("--rgb-shift-y", type=int, default=0, help="Shift Y (pix) post-alineacion.")
    p.add_argument("--stride", type=int, default=3, help="Submuestreo (default: 3).")
    p.add_argument("--conf-min", type=int, default=None, help="Confianza minima (si existe).")
    p.add_argument("--z-min", type=float, default=0.0, help="Filtrar Z <= z_min.")
    p.add_argument("--max-points", type=int, default=0, help="Limitar puntos (0 = sin limite).")
    p.add_argument("--seed", type=int, default=0, help="Seed para subsample.")
    p.add_argument("--out", type=Path, help="Guardar HTML.")
    p.add_argument("--no-show", action="store_true", help="No abrir visor.")
    p.add_argument("--flip-x", action="store_true", help="Invertir eje X.")
    p.add_argument(
        "--flip-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invertir eje Y (default: True).",
    )
    p.add_argument(
        "--flip-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invertir eje Z (default: True).",
    )
    p.add_argument("--auto-flip-z", action="store_true", help="Invierte Z si la mediana es negativa.")
    p.add_argument("--save-aligned", type=Path, help="Guardar RGB alineado para inspeccion.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    xyz, conf = load_npz_xyz_conf(args.npz)
    rgb_path = resolve_rgb_path(args.rgb)
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise SystemExit(f"No pude leer RGB: {rgb_path}")

    aligned_bgr = align_rgb_to_tof(
        rgb_bgr,
        tof_shape=xyz.shape[:2],
        shift_xy=(args.rgb_shift_x, args.rgb_shift_y),
    )
    if args.save_aligned:
        cv2.imwrite(str(args.save_aligned), aligned_bgr)

    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    pts, cols = build_points_with_colors(
        xyz=xyz,
        rgb_aligned=aligned_rgb,
        confidence=conf,
        conf_min=args.conf_min,
        z_min=args.z_min,
        stride=max(1, args.stride),
        max_points=args.max_points,
        seed=args.seed,
    )

    if pts.size == 0:
        raise SystemExit("No hay puntos validos. Ajusta conf_min / z_min / stride.")

    pts = apply_axis_flips(
        pts,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
        auto_flip_z=args.auto_flip_z,
    )

    viewer = PointCloudViewerPlotly(title="Point Cloud + RGB (Plotly)")
    viewer.set_points(pts, colors_rgb=cols)

    if args.out:
        viewer.save_html(args.out)
        print(f"[OK] HTML guardado en: {args.out}")

    if not args.no_show:
        viewer.show()


if __name__ == "__main__":
    main()
