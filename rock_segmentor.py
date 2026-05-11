# rock_segmentor.py
# Pipeline de segmentación de rocas en tiempo real para cámara Basler ToF 101.
#
# Pipeline por frame:
#   xyz (H,W,3)  →  outlier removal  →  RANSAC ground  →  features globales
#   →  XGBoost prob_roca  →  [si prob > umbral]  DBSCAN clusters  →  OBBs

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import joblib
from scipy.spatial import KDTree

log = logging.getLogger("rock_segmentor")

FEATURES_ORDER = ["altura_norm", "densidad", "curvatura", "normal_z", "intensidad"]

# ---------------------------------------------------------------------------
# Resultado por nube
# ---------------------------------------------------------------------------

@dataclass
class RockCluster:
    cluster_id:  int
    prob_roca:   float
    n_points:    int
    obb_center:  np.ndarray          # (3,) float32
    obb_extent:  np.ndarray          # (3,) float32 — semiejes
    obb_corners: np.ndarray          # (8,3) float32
    obb_R:       np.ndarray          # (3,3) float32 — rotación


@dataclass
class SegResult:
    timestamp:     float
    prob_roca:     float              # probabilidad global de la nube
    is_rock:       bool
    ground_pts:    Optional[np.ndarray] = None   # (N,3)
    object_pts:    Optional[np.ndarray] = None   # (M,3)
    object_labels: Optional[np.ndarray] = None   # (M,) int — label DBSCAN por punto (-1=ruido)
    clusters:      List[RockCluster] = field(default_factory=list)
    elapsed_ms:    float = 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SegConfig:
    # Filtro de confianza
    conf_min: int = 200

    # Remoción de outliers estadísticos
    outlier_nb:    int   = 20
    outlier_std:   float = 2.0

    # RANSAC plano suelo
    ransac_dist:   float = 0.05    # metros
    ransac_n:      int   = 3
    ransac_iter:   int   = 1000

    # Submuestreo de objetos antes de DBSCAN
    max_object_pts: int  = 80_000

    # DBSCAN
    dbscan_knn:       int   = 15    # vecinos para estimar eps
    dbscan_eps_pct:   float = 95.0  # percentil de distancias k-NN
    dbscan_min_pts:   int   = 50

    # Umbral de clasificación
    prob_threshold: float = 0.5

    # Modelos serializados
    model_path:  str = "models/rock_classifier.joblib"
    scaler_path: str = "models/rock_scaler.joblib"

    # Escala XYZ (blaze devuelve mm → m)
    xyz_scale: float = 0.001


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class RockSegmentor:
    def __init__(self, cfg: SegConfig = SegConfig()):
        self.cfg = cfg
        self._clf    = None
        self._scaler = None
        self._load_model()

    def _load_model(self) -> None:
        cfg = self.cfg
        if os.path.exists(cfg.model_path) and os.path.exists(cfg.scaler_path):
            self._clf    = joblib.load(cfg.model_path)
            self._scaler = joblib.load(cfg.scaler_path)
            log.info("Modelo cargado: %s", cfg.model_path)
        else:
            log.warning(
                "No se encontró modelo en %s. Usando umbral de altura como proxy.",
                cfg.model_path
            )

    def process(
        self,
        xyz_hw3:    np.ndarray,
        confidence: Optional[np.ndarray] = None,
        intensity:  Optional[np.ndarray] = None,
    ) -> SegResult:
        t0 = time.perf_counter()
        cfg = self.cfg

        # 1. Construir nube válida
        pts, intens_vals = self._build_cloud(xyz_hw3, confidence, intensity)
        if pts.shape[0] < 100:
            return SegResult(timestamp=time.time(), prob_roca=0.0, is_rock=False,
                             elapsed_ms=(time.perf_counter() - t0) * 1000)

        # 2. Outlier removal estadístico
        pts, intens_vals = self._remove_outliers(pts, intens_vals)

        # 3. RANSAC — separar suelo de objetos
        ground_pts, object_pts, obj_intens = self._ransac_ground(pts, intens_vals)

        # 4. Features globales (sobre puntos de objetos)
        feats = self._compute_features(object_pts, obj_intens)

        # 5. Clasificación XGBoost
        prob_roca = self._classify(feats, object_pts)
        is_rock   = prob_roca >= cfg.prob_threshold

        # 6. DBSCAN solo si es nube de roca
        clusters: List[RockCluster] = []
        object_labels: Optional[np.ndarray] = None
        if is_rock and object_pts.shape[0] >= cfg.dbscan_min_pts:
            clusters, object_labels = self._dbscan_and_obb(object_pts, prob_roca)

        elapsed = (time.perf_counter() - t0) * 1000
        return SegResult(
            timestamp     = time.time(),
            prob_roca     = float(prob_roca),
            is_rock       = is_rock,
            ground_pts    = ground_pts,
            object_pts    = object_pts,
            object_labels = object_labels,
            clusters      = clusters,
            elapsed_ms    = elapsed,
        )

    # ------------------------------------------------------------------
    # Pasos internos
    # ------------------------------------------------------------------

    def _build_cloud(
        self,
        xyz_hw3:    np.ndarray,
        confidence: Optional[np.ndarray],
        intensity:  Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = xyz_hw3.reshape(-1, 3).astype(np.float32) * self.cfg.xyz_scale

        valid = np.isfinite(pts).all(axis=1)
        valid &= (np.linalg.norm(pts, axis=1) > 1e-4)

        if confidence is not None:
            valid &= (confidence.reshape(-1) >= self.cfg.conf_min)

        pts = pts[valid]
        intens = intensity.reshape(-1)[valid].astype(np.float32) if intensity is not None else None
        return pts, intens

    def _remove_outliers(
        self,
        pts:    np.ndarray,
        intens: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        _, mask = pcd.remove_statistical_outlier(
            nb_neighbors=self.cfg.outlier_nb,
            std_ratio=self.cfg.outlier_std,
        )
        mask = np.asarray(mask)
        cleaned_intens = intens[mask] if intens is not None else None
        return pts[mask], cleaned_intens

    def _ransac_ground(
        self,
        pts:    np.ndarray,
        intens: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        _, inliers = pcd.segment_plane(
            distance_threshold=self.cfg.ransac_dist,
            ransac_n=self.cfg.ransac_n,
            num_iterations=self.cfg.ransac_iter,
        )
        inliers = np.asarray(inliers)
        mask_obj = np.ones(len(pts), dtype=bool)
        mask_obj[inliers] = False

        ground_pts = pts[inliers]
        object_pts = pts[mask_obj]
        obj_intens = intens[mask_obj] if intens is not None else None

        # Subsample si hay demasiados puntos de objetos
        if object_pts.shape[0] > self.cfg.max_object_pts:
            idx = np.random.choice(object_pts.shape[0], self.cfg.max_object_pts, replace=False)
            object_pts = object_pts[idx]
            if obj_intens is not None:
                obj_intens = obj_intens[idx]

        return ground_pts, object_pts, obj_intens

    def _compute_features(
        self,
        pts:    np.ndarray,
        intens: Optional[np.ndarray],
    ) -> dict:
        if pts.shape[0] < 10:
            return {f: 0.0 for f in FEATURES_ORDER}

        tree = KDTree(pts)
        k = 10

        # Distancias a k vecinos más cercanos (sin el punto mismo)
        dists, _ = tree.query(pts, k=k + 1)
        dists = dists[:, 1:]   # excluir distancia 0 a sí mismo

        z = pts[:, 2]
        z_min, z_max = z.min(), z.max()
        z_range = max(z_max - z_min, 1e-6)

        altura_norm = float(np.mean((z - z_min) / z_range))
        densidad    = float(np.mean(1.0 / (dists[:, 0] + 1e-6)))
        curvatura   = float(np.std(dists))

        # normal_z: variación de Z en vecindad local normalizada por rango total
        _, knn_idx = tree.query(pts, k=min(6, pts.shape[0]))
        z_neighbors = z[knn_idx]   # (N, k)
        normal_z = float(np.mean(np.std(z_neighbors, axis=1)) / z_range)

        if intens is not None:
            i_min, i_max = intens.min(), intens.max()
            i_range = max(i_max - i_min, 1.0)
            intensidad = float(np.mean((intens - i_min) / i_range))
        else:
            # fallback: intensidad ≈ altura_norm (como en features.py del ML)
            intensidad = altura_norm

        return {
            "altura_norm": altura_norm,
            "densidad":    densidad,
            "curvatura":   curvatura,
            "normal_z":    normal_z,
            "intensidad":  intensidad,
        }

    def _classify(self, feats: dict, object_pts: np.ndarray) -> float:
        if self._clf is None or self._scaler is None:
            # Sin modelo: proxy simple — si hay muchos puntos altos → roca
            z = object_pts[:, 2]
            altura_media = feats["altura_norm"]
            return float(np.clip(altura_media * 2.0, 0.0, 1.0))

        vec = np.array([[feats[k] for k in FEATURES_ORDER]], dtype=np.float32)
        vec_sc = self._scaler.transform(vec)
        prob = float(self._clf.predict_proba(vec_sc)[0, 1])
        return prob

    def _dbscan_and_obb(
        self,
        object_pts: np.ndarray,
        prob_roca:  float,
    ) -> Tuple[List[RockCluster], np.ndarray]:
        """Retorna (clusters, labels_por_punto) donde labels=-1 es ruido."""
        cfg = self.cfg

        # Estimar eps desde percentil de distancias k-NN
        k = min(cfg.dbscan_knn, object_pts.shape[0] - 1)
        tree = KDTree(object_pts)
        dists, _ = tree.query(object_pts, k=k + 1)
        eps = float(np.percentile(dists[:, k], cfg.dbscan_eps_pct))
        eps = max(eps, 0.01)   # mínimo 1 cm

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_pts.astype(np.float64))

        labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=cfg.dbscan_min_pts, print_progress=False)
        )

        clusters: List[RockCluster] = []
        unique_labels = np.unique(labels)

        for cid in unique_labels:
            if cid < 0:
                continue
            mask = labels == cid
            c_pts = object_pts[mask]

            c_pcd = o3d.geometry.PointCloud()
            c_pcd.points = o3d.utility.Vector3dVector(c_pts.astype(np.float64))

            try:
                obb = c_pcd.get_oriented_bounding_box()
                corners = np.asarray(obb.get_box_points(), dtype=np.float32)
                center  = np.asarray(obb.center, dtype=np.float32)
                extent  = np.asarray(obb.extent, dtype=np.float32)
                R       = np.asarray(obb.R, dtype=np.float32)
            except Exception:
                continue

            clusters.append(RockCluster(
                cluster_id  = int(cid),
                prob_roca   = prob_roca,
                n_points    = int(mask.sum()),
                obb_center  = center,
                obb_extent  = extent,
                obb_corners = corners,
                obb_R       = R,
            ))

        return clusters, labels
