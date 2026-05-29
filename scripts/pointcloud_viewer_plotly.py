import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go


class PointCloudViewerPlotly:
    def __init__(self, title: str = "Point Cloud", width: int = 1100, height: int = 800,
                 point_size: float = 2.0, show_legend: bool = False) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.point_size = point_size
        self.show_legend = show_legend
        self._fig: Optional[go.Figure] = None

    def set_points(self, points_xyz: np.ndarray, colors_rgb: Optional[np.ndarray] = None) -> None:
        pts = np.asarray(points_xyz, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_xyz debe ser Nx3, recibí {pts.shape}")

        if colors_rgb is None:
            colors = "rgb(30,144,255)"
        else:
            cols = np.asarray(colors_rgb)
            if cols.ndim != 2 or cols.shape[1] != 3 or cols.shape[0] != pts.shape[0]:
                raise ValueError(f"colors_rgb debe ser Nx3 y coincidir con puntos, recibí {cols.shape}")
            if np.issubdtype(cols.dtype, np.floating):
                cols = np.clip(cols, 0.0, 1.0) * 255.0
            cols = np.clip(cols, 0, 255).astype(np.uint8)
            colors = [f"rgb({r},{g},{b})" for r, g, b in cols]

        self._fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=self.point_size, color=colors, opacity=1.0),
                    name="points",
                )
            ]
        )
        self._fig.update_layout(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=self.show_legend,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                aspectmode="data",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
        )

    def show(self) -> None:
        if self._fig is None:
            raise RuntimeError("Primero llama set_points().")
        self._fig.show()

    def save_html(self, path: Path) -> None:
        if self._fig is None:
            raise RuntimeError("Primero llama set_points().")
        self._fig.write_html(str(path))


def load_xyz_from_npz(
    npz_path: Path,
    stride: int = 2,
    conf_min: Optional[int] = None,
    z_min: float = 0.0,
    max_points: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(str(npz_path))
    if "xyz" not in data.files:
        raise ValueError(f"NPZ sin key 'xyz': {npz_path}")

    xyz = data["xyz"]
    conf = data["confidence"] if "confidence" in data.files else None

    xyz_s = xyz[::stride, ::stride, :]
    conf_s = conf[::stride, ::stride] if conf is not None else None

    pts = xyz_s.reshape(-1, 3).astype(np.float32)
    mask = np.isfinite(pts).all(axis=1)
    mask &= (pts[:, 2] > z_min)
    if conf_s is not None and conf_min is not None:
        mask &= (conf_s.reshape(-1) >= conf_min)

    pts = pts[mask]

    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    return pts, conf_s


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
    p = argparse.ArgumentParser(description="Visor Plotly de nube de puntos (Step 1).")
    p.add_argument("--npz", type=Path, required=True, help="Ruta a NPZ con 'xyz'.")
    p.add_argument("--stride", type=int, default=3, help="Submuestreo (default: 3).")
    p.add_argument("--conf-min", type=int, default=None, help="Confianza mínima (si existe).")
    p.add_argument("--z-min", type=float, default=0.0, help="Filtrar Z <= z_min.")
    p.add_argument("--max-points", type=int, default=0, help="Limitar puntos (0 = sin límite).")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pts, _ = load_xyz_from_npz(
        npz_path=args.npz,
        stride=max(1, args.stride),
        conf_min=args.conf_min,
        z_min=args.z_min,
        max_points=args.max_points,
        seed=args.seed,
    )

    if pts.size == 0:
        raise SystemExit("No hay puntos válidos. Ajusta conf_min / z_min / stride.")

    pts = apply_axis_flips(
        pts,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
        auto_flip_z=args.auto_flip_z,
    )

    viewer = PointCloudViewerPlotly(title="Point Cloud (Plotly)")
    viewer.set_points(pts)

    if args.out:
        viewer.save_html(args.out)
        print(f"[OK] HTML guardado en: {args.out}")

    if not args.no_show:
        viewer.show()


if __name__ == "__main__":
    main()
