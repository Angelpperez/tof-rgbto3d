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
    centroid:    np.ndarray          # (3,) float32, coord. camara
    centroid_view: np.ndarray        # (3,) float32, coord. visor
    base_center: np.ndarray          # (3,) float32, centro en plano suelo
    base_center_view: np.ndarray     # (3,) float32, coord. visor
    height_above_ground: float
    obb_center:  np.ndarray          # (3,) float32
    obb_extent:  np.ndarray          # (3,) float32
    obb_corners: np.ndarray          # (8,3) float32
    obb_R:       np.ndarray          # (3,3) float32 — rotación
    top_face_center:      np.ndarray # (3,) float32, base + normal suelo * altura
    top_face_center_view: np.ndarray # (3,) float32, coord. visor
    candidate_score: float = 0.0
    tracking_status: str = "selected"
    match_distance: float = 0.0


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
    tracking_status: str = "none"


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
    ransac_dist:        float = 0.03    # metros — tolerancia al ajustar el plano
    ransac_n:           int   = 3
    ransac_iter:        int   = 1000
    min_height_above:   float = 0.07    # metros — ignora relieve/grizzly bajo sobre el suelo

    # Submuestreo de objetos antes de DBSCAN
    max_object_pts: int  = 80_000

    # DBSCAN
    dbscan_knn:       int   = 15    # vecinos para estimar eps
    dbscan_eps_pct:   float = 95.0  # percentil de distancias k-NN
    dbscan_eps_min:   float = 0.03
    dbscan_min_pts:   int   = 50

    # Tracking temporal: evita saltar al picaroca/pierna cuando entra en escena
    tracking_enabled: bool = True
    track_max_center_jump: float = 0.35
    track_max_extent_ratio: float = 2.5
    track_hold_frames: int = 90

    # Priors geometricos para preferir rocas y penalizar intrusos largos/altos
    rock_min_height: float = 0.03
    rock_max_height: float = 1.30
    rock_max_aspect: float = 12.0
    rock_max_footprint: float = 2.50

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
        self._last_cluster: Optional[RockCluster] = None
        self._missed_track_frames = 0
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
        ground_pts, object_pts, obj_intens, obj_heights, ground_normal = self._ransac_ground(
            pts, intens_vals
        )

        # 4. Features globales (sobre puntos de objetos)
        feats = self._compute_features(object_pts, obj_intens)

        # 5. Clasificación XGBoost
        prob_roca = self._classify(feats, object_pts)
        is_rock   = prob_roca >= cfg.prob_threshold

        # 6. DBSCAN solo si es nube de roca
        clusters: List[RockCluster] = []
        object_labels: Optional[np.ndarray] = None
        tracking_status = "not_rock"
        if is_rock and object_pts.shape[0] >= cfg.dbscan_min_pts:
            clusters, object_labels, tracking_status = self._dbscan_and_obb(
                object_pts, obj_heights, ground_normal, prob_roca
            )
        elif cfg.tracking_enabled and self._last_cluster is not None:
            selected, tracking_status = self._select_tracked_cluster([])
            clusters = [selected] if selected is not None else []

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
            tracking_status = tracking_status,
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
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        plane_model, _ = pcd.segment_plane(
            distance_threshold=self.cfg.ransac_dist,
            ransac_n=self.cfg.ransac_n,
            num_iterations=self.cfg.ransac_iter,
        )

        # Distancia firmada de cada punto al plano ax+by+cz+d=0
        # Positivo = por encima del plano (lado de la normal)
        a, b, c, d = plane_model
        norm = float(np.sqrt(a*a + b*b + c*c))
        ground_normal = np.array([a, b, c], dtype=np.float32) / norm
        plane_offset = float(d) / norm
        signed_dist = pts @ ground_normal + plane_offset
        if signed_dist.mean() < 0:
            ground_normal = -ground_normal
            plane_offset = -plane_offset
            signed_dist = -signed_dist

        mask_obj = signed_dist > self.cfg.min_height_above

        ground_pts = pts[~mask_obj]
        object_pts = pts[mask_obj]
        obj_intens = intens[mask_obj] if intens is not None else None
        obj_heights = signed_dist[mask_obj].astype(np.float32)

        log.debug("RANSAC: suelo=%d obj=%d (min_height=%.3fm)",
                  ground_pts.shape[0], object_pts.shape[0], self.cfg.min_height_above)

        # Subsample si hay demasiados puntos de objetos
        if object_pts.shape[0] > self.cfg.max_object_pts:
            idx = np.random.choice(object_pts.shape[0], self.cfg.max_object_pts, replace=False)
            object_pts = object_pts[idx]
            obj_heights = obj_heights[idx]
            if obj_intens is not None:
                obj_intens = obj_intens[idx]

        return ground_pts, object_pts, obj_intens, obj_heights, ground_normal

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
        object_heights: np.ndarray,
        ground_normal: np.ndarray,
        prob_roca:  float,
    ) -> Tuple[List[RockCluster], np.ndarray, str]:
        """Retorna solo la roca trackeada y labels para todos los objetos."""
        cfg = self.cfg

        # Estimar eps desde percentil de distancias k-NN
        k = min(cfg.dbscan_knn, object_pts.shape[0] - 1)
        tree = KDTree(object_pts)
        dists, _ = tree.query(object_pts, k=k + 1)
        eps = float(np.percentile(dists[:, k], cfg.dbscan_eps_pct))
        eps = max(eps, cfg.dbscan_eps_min)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_pts.astype(np.float64))

        labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=cfg.dbscan_min_pts, print_progress=False)
        )

        valid_labels = labels[labels >= 0]
        if valid_labels.size == 0:
            selected, status = self._select_tracked_cluster([])
            return ([selected] if selected is not None else []), labels.astype(np.int32), status

        unique_labels = np.unique(valid_labels)
        candidates: List[RockCluster] = []
        for cid in unique_labels:
            mask = labels == cid
            if int(mask.sum()) < cfg.dbscan_min_pts:
                continue
            cluster = self._build_cluster_candidate(
                int(cid),
                object_pts[mask],
                object_heights[mask],
                ground_normal,
                prob_roca,
            )
            if cluster is not None:
                candidates.append(cluster)

        selected, status = self._select_tracked_cluster(candidates)
        return ([selected] if selected is not None else []), labels.astype(np.int32), status

    def _build_cluster_candidate(
        self,
        cid: int,
        c_pts: np.ndarray,
        c_heights: np.ndarray,
        ground_normal: np.ndarray,
        prob_roca: float,
    ) -> Optional[RockCluster]:
        try:
            centroid = np.mean(c_pts, axis=0).astype(np.float32)
            centroid_view = self._flip_y_point(centroid)
            center, extent, R, corners, base_center, top_center, height = self._ground_aligned_obb(
                c_pts, c_heights, ground_normal
            )
            base_center_view = self._flip_y_point(base_center)
            top_center_view = self._flip_y_point(top_center)
        except Exception:
            return None

        cluster = RockCluster(
            cluster_id  = cid,
            prob_roca   = prob_roca,
            n_points    = int(c_pts.shape[0]),
            centroid    = centroid,
            centroid_view = centroid_view,
            base_center = base_center,
            base_center_view = base_center_view,
            height_above_ground = height,
            obb_center  = center,
            obb_extent  = extent,
            obb_corners = corners,
            obb_R       = R,
            top_face_center      = top_center,
            top_face_center_view = top_center_view,
        )
        if not self._is_plausible_rock_candidate(cluster):
            return None
        cluster.candidate_score = self._rock_candidate_score(cluster)
        return cluster

    def _is_plausible_rock_candidate(self, cluster: RockCluster) -> bool:
        cfg = self.cfg
        footprint_a = max(float(cluster.obb_extent[0]), 1e-6)
        footprint_b = max(float(cluster.obb_extent[1]), 1e-6)
        length = max(footprint_a, footprint_b)
        width = min(footprint_a, footprint_b)
        aspect = length / max(width, 1e-6)
        footprint = length * width
        height = float(cluster.height_above_ground)

        return (
            height >= cfg.rock_min_height
            and height <= cfg.rock_max_height
            and aspect <= cfg.rock_max_aspect
            and footprint <= cfg.rock_max_footprint
        )

    def _rock_candidate_score(self, cluster: RockCluster) -> float:
        cfg = self.cfg
        footprint_a = max(float(cluster.obb_extent[0]), 1e-6)
        footprint_b = max(float(cluster.obb_extent[1]), 1e-6)
        length = max(footprint_a, footprint_b)
        width = min(footprint_a, footprint_b)
        height = max(float(cluster.height_above_ground), 1e-6)
        aspect = length / max(width, 1e-6)
        footprint = length * width

        score = float(np.log1p(cluster.n_points))
        if height < cfg.rock_min_height:
            score -= (cfg.rock_min_height - height) * 20.0
        if height > cfg.rock_max_height:
            score -= (height - cfg.rock_max_height) * 8.0
        if aspect > cfg.rock_max_aspect:
            score -= (aspect - cfg.rock_max_aspect) * 0.35
        if footprint > cfg.rock_max_footprint:
            score -= (footprint - cfg.rock_max_footprint) * 2.0
        return score

    def _select_tracked_cluster(
        self,
        candidates: List[RockCluster],
    ) -> Tuple[Optional[RockCluster], str]:
        cfg = self.cfg

        if not candidates:
            if self._last_cluster is not None and self._should_hold_track():
                return self._hold_last_cluster(), "hold"
            self._last_cluster = None
            self._missed_track_frames = 0
            return None, "lost"

        if not cfg.tracking_enabled or self._last_cluster is None:
            selected = max(candidates, key=lambda c: c.candidate_score)
            self._commit_track(selected, "init")
            return selected, "init"

        prev = self._last_cluster
        matches: list[Tuple[float, RockCluster]] = []
        for candidate in candidates:
            center_dist = float(np.linalg.norm(candidate.base_center - prev.base_center))
            extent_ratio = self._max_extent_ratio(candidate.obb_extent, prev.obb_extent)
            candidate.match_distance = center_dist

            if center_dist > cfg.track_max_center_jump:
                continue
            if extent_ratio > cfg.track_max_extent_ratio:
                continue

            tracking_score = (
                candidate.candidate_score
                - 6.0 * (center_dist / max(cfg.track_max_center_jump, 1e-6))
                - 2.0 * np.log(max(extent_ratio, 1.0))
            )
            matches.append((float(tracking_score), candidate))

        if matches:
            selected = max(matches, key=lambda item: item[0])[1]
            self._commit_track(selected, "tracked")
            return selected, "tracked"

        if self._should_hold_track():
            return self._hold_last_cluster(), "hold"

        selected = max(candidates, key=lambda c: c.candidate_score)
        self._commit_track(selected, "reacquired")
        return selected, "reacquired"

    def _commit_track(self, cluster: RockCluster, status: str) -> None:
        cluster.tracking_status = status
        self._last_cluster = cluster
        self._missed_track_frames = 0

    def _should_hold_track(self) -> bool:
        self._missed_track_frames += 1
        return (
            self._last_cluster is not None
            and self._missed_track_frames <= self.cfg.track_hold_frames
        )

    def _hold_last_cluster(self) -> RockCluster:
        assert self._last_cluster is not None
        self._last_cluster.tracking_status = "hold"
        return self._last_cluster

    @staticmethod
    def _max_extent_ratio(a: np.ndarray, b: np.ndarray) -> float:
        a = np.maximum(a.astype(np.float32), 0.03)
        b = np.maximum(b.astype(np.float32), 0.03)
        ratios = np.maximum(a / b, b / a)
        return float(np.max(ratios))

    @staticmethod
    def _ground_aligned_obb(
        pts: np.ndarray,
        heights: np.ndarray,
        ground_normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        n = ground_normal.astype(np.float32)
        n /= max(float(np.linalg.norm(n)), 1e-6)

        base_pts = pts - heights.reshape(-1, 1).astype(np.float32) * n
        base_mean = np.mean(base_pts, axis=0).astype(np.float32)

        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(ref @ n)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        u = ref - float(ref @ n) * n
        u /= max(float(np.linalg.norm(u)), 1e-6)
        v = np.cross(n, u).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-6)

        centered = base_pts - base_mean
        coords = np.column_stack((centered @ u, centered @ v)).astype(np.float32)
        if coords.shape[0] >= 3:
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
            u = (u * principal[0] + v * principal[1]).astype(np.float32)
            u /= max(float(np.linalg.norm(u)), 1e-6)
            v = np.cross(n, u).astype(np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-6)
            coords = np.column_stack((centered @ u, centered @ v)).astype(np.float32)

        xy_min = coords.min(axis=0)
        xy_max = coords.max(axis=0)
        xy_mid = (xy_min + xy_max) * 0.5
        width, depth = (xy_max - xy_min).astype(np.float32)

        height = max(float(np.max(heights)), 0.001)
        base_center = (base_mean + u * xy_mid[0] + v * xy_mid[1]).astype(np.float32)
        top_center = (base_center + n * height).astype(np.float32)
        center = (base_center + n * (height * 0.5)).astype(np.float32)
        extent = np.array([max(float(width), 0.001), max(float(depth), 0.001), height], dtype=np.float32)
        R = np.column_stack((u, v, n)).astype(np.float32)
        if np.linalg.det(R.astype(np.float64)) < 0:
            R[:, 1] = -R[:, 1]

        corners = RockSegmentor._obb_corners(center, extent, R)
        return center, extent, R, corners, base_center, top_center, float(height)

    @staticmethod
    def _obb_corners(center: np.ndarray, extent: np.ndarray, R: np.ndarray) -> np.ndarray:
        half = extent.astype(np.float32) * 0.5
        local = np.array([
            [-half[0], -half[1], -half[2]],
            [ half[0], -half[1], -half[2]],
            [ half[0],  half[1], -half[2]],
            [-half[0],  half[1], -half[2]],
            [-half[0], -half[1],  half[2]],
            [ half[0], -half[1],  half[2]],
            [ half[0],  half[1],  half[2]],
            [-half[0],  half[1],  half[2]],
        ], dtype=np.float32)
        return (center.reshape(1, 3) + local @ R.T).astype(np.float32)

    @staticmethod
    def _flip_y_point(point: np.ndarray) -> np.ndarray:
        out = point.astype(np.float32).copy()
        out[1] = -out[1]
        return out
