from __future__ import annotations

import asyncio
import glob
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger("segmentation_api")


class LatestSegmentationStore:
    def __init__(self, source: str = "idle") -> None:
        self._lock = threading.Lock()
        self._result: Optional[Any] = None
        self._source = source
        self._frame: Optional[str] = None
        self._frame_id = 0
        self._updated_at = 0.0

    def set_result(self, result: Any, source: str, frame: Optional[str] = None) -> None:
        with self._lock:
            self._result = result
            self._source = source
            self._frame = frame
            self._frame_id += 1
            self._updated_at = time.time()

    def set_source(self, source: str) -> None:
        with self._lock:
            self._source = source

    def get_result(self) -> Optional[Any]:
        with self._lock:
            return self._result

    def status(self) -> dict:
        with self._lock:
            return {
                "source": self._source,
                "frame": self._frame,
                "frame_id": self._frame_id,
                "updated_at": self._updated_at or None,
                "has_result": self._result is not None,
            }


def _vec3(value: Optional[np.ndarray]) -> Optional[dict]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])}


def _vec_list(value: Optional[np.ndarray]) -> Optional[list]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    return arr.tolist()


def _count_points(value: Optional[np.ndarray]) -> int:
    return 0 if value is None else int(value.shape[0])


def _cluster_payload(cluster) -> dict:
    extent = np.asarray(cluster.obb_extent, dtype=np.float64).reshape(3)
    length = float(max(extent[0], extent[1]))
    width = float(min(extent[0], extent[1]))
    height = float(extent[2])
    return {
        "cluster_id": int(cluster.cluster_id),
        "prob_roca": float(cluster.prob_roca),
        "n_points": int(cluster.n_points),
        "tracking_status": str(getattr(cluster, "tracking_status", "selected")),
        "match_distance_m": float(getattr(cluster, "match_distance", 0.0)),
        "height_above_ground_m": float(cluster.height_above_ground),
        "top_face_center_m": _vec3(cluster.top_face_center),
        "centroid_m": _vec3(cluster.centroid),
        "base_center_m": _vec3(cluster.base_center),
        "obb_center_m": _vec3(cluster.obb_center),
        "obb_extent_m": _vec3(cluster.obb_extent),
        "obb_corners_m": _vec_list(cluster.obb_corners),
        "dimensions_m": {
            "length": length,
            "width": width,
            "height": height,
        },
    }


def seg_result_payload(result: Optional[Any], status: Optional[dict] = None) -> dict:
    status = status or {}
    if result is None:
        return {
            "available": False,
            "source": status.get("source", "idle"),
            "frame": status.get("frame"),
            "frame_id": status.get("frame_id", 0),
            "updated_at": status.get("updated_at"),
            "message": "No hay segmentacion disponible todavia.",
        }

    clusters = [_cluster_payload(cluster) for cluster in result.clusters]
    return {
        "available": True,
        "source": status.get("source", "live"),
        "frame": status.get("frame"),
        "frame_id": status.get("frame_id", 0),
        "timestamp": float(result.timestamp),
        "age_s": max(0.0, time.time() - float(result.timestamp)),
        "updated_at": status.get("updated_at"),
        "prob_roca": float(result.prob_roca),
        "is_rock": bool(result.is_rock),
        "tracking_status": str(result.tracking_status),
        "elapsed_ms": float(result.elapsed_ms),
        "counts": {
            "ground_pts": _count_points(result.ground_pts),
            "object_pts": _count_points(result.object_pts),
            "clusters": len(result.clusters),
        },
        "target": clusters[0] if clusters else None,
        "clusters": clusters,
    }


def target_payload(result: Optional[Any], status: Optional[dict] = None) -> dict:
    payload = seg_result_payload(result, status)
    if not payload.get("available"):
        return payload
    target = payload.get("target")
    if target is None:
        return {
            "available": False,
            "source": payload.get("source"),
            "frame": payload.get("frame"),
            "frame_id": payload.get("frame_id"),
            "timestamp": payload.get("timestamp"),
            "is_rock": payload.get("is_rock"),
            "tracking_status": payload.get("tracking_status"),
            "message": "No hay target de roca en la segmentacion actual.",
        }
    return {
        "available": True,
        "source": payload.get("source"),
        "frame": payload.get("frame"),
        "frame_id": payload.get("frame_id"),
        "timestamp": payload.get("timestamp"),
        "is_rock": payload.get("is_rock"),
        "tracking_status": payload.get("tracking_status"),
        "top_face_center_m": target["top_face_center_m"],
        "dimensions_m": target["dimensions_m"],
        "cluster_id": target["cluster_id"],
        "n_points": target["n_points"],
        "prob_roca": target["prob_roca"],
    }


def create_app(
    get_result: Callable[[], Optional[Any]],
    get_status: Optional[Callable[[], dict]] = None,
    title: str = "3dcamera Segmentation API",
):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse

    app = FastAPI(title=title)

    origins = [
        origin.strip()
        for origin in os.getenv("SEG_API_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def status() -> dict:
        return get_status() if get_status is not None else {}

    @app.get("/health")
    def health() -> dict:
        current_status = status()
        return {
            "status": "ok",
            "has_result": get_result() is not None,
            **current_status,
        }

    @app.get("/segmentation/latest")
    def latest() -> dict:
        return seg_result_payload(get_result(), status())

    @app.get("/segmentation/target")
    def target() -> dict:
        return target_payload(get_result(), status())

    @app.websocket("/segmentation/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        last_timestamp = object()
        try:
            while True:
                result = get_result()
                current_timestamp = getattr(result, "timestamp", None)
                if current_timestamp != last_timestamp:
                    await websocket.send_json(seg_result_payload(result, status()))
                    last_timestamp = current_timestamp
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return

    @app.get("/viewer", response_class=HTMLResponse)
    def viewer() -> str:
        return _viewer_html()

    return app


def _viewer_html() -> str:
    return """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3dcamera Segmentation</title>
  <style>
    :root { color-scheme: dark; font-family: Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #0d1117; color: #d7dee8; }
    main { max-width: 980px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 18px; font-size: 24px; font-weight: 650; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card { border: 1px solid #273241; border-radius: 8px; padding: 16px; background: #121821; }
    .label { color: #8ea0b8; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { margin-top: 6px; font-size: 26px; font-variant-numeric: tabular-nums; }
    pre { overflow: auto; padding: 16px; border-radius: 8px; background: #070a0f; border: 1px solid #273241; }
    .ok { color: #37d67a; }
    .warn { color: #ffcc33; }
  </style>
</head>
<body>
<main>
  <h1>3dcamera segmentation</h1>
  <div class="grid">
    <section class="card"><div class="label">Estado</div><div id="state" class="value warn">sin datos</div></section>
    <section class="card"><div class="label">Track</div><div id="track" class="value">-</div></section>
    <section class="card"><div class="label">X m</div><div id="x" class="value">-</div></section>
    <section class="card"><div class="label">Y m</div><div id="y" class="value">-</div></section>
    <section class="card"><div class="label">Z m</div><div id="z" class="value">-</div></section>
    <section class="card"><div class="label">Dimensiones m</div><div id="dims" class="value">-</div></section>
  </div>
  <h2>Payload</h2>
  <pre id="payload">{}</pre>
</main>
<script>
const els = {
  state: document.getElementById('state'),
  track: document.getElementById('track'),
  x: document.getElementById('x'),
  y: document.getElementById('y'),
  z: document.getElementById('z'),
  dims: document.getElementById('dims'),
  payload: document.getElementById('payload')
};
function fmt(v) { return Number.isFinite(v) ? v.toFixed(3) : '-'; }
function render(data) {
  els.payload.textContent = JSON.stringify(data, null, 2);
  if (!data.available || !data.target) {
    els.state.textContent = 'sin target';
    els.state.className = 'value warn';
    els.track.textContent = data.tracking_status || '-';
    els.x.textContent = els.y.textContent = els.z.textContent = els.dims.textContent = '-';
    return;
  }
  const p = data.target.top_face_center_m;
  const d = data.target.dimensions_m;
  els.state.textContent = data.is_rock ? 'roca' : 'no roca';
  els.state.className = data.is_rock ? 'value ok' : 'value warn';
  els.track.textContent = data.tracking_status || '-';
  els.x.textContent = fmt(p.x);
  els.y.textContent = fmt(p.y);
  els.z.textContent = fmt(p.z);
  els.dims.textContent = `${fmt(d.length)} x ${fmt(d.width)} x ${fmt(d.height)}`;
}
function start() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/segmentation/stream`);
  ws.onmessage = event => render(JSON.parse(event.data));
  ws.onclose = () => setTimeout(start, 1000);
  ws.onerror = () => ws.close();
}
start();
</script>
</body>
</html>"""


def run_api_server(
    get_result: Callable[[], Optional[Any]],
    get_status: Optional[Callable[[], dict]] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    title: str = "3dcamera Segmentation API",
) -> None:
    import uvicorn

    app = create_app(get_result, get_status=get_status, title=title)
    uvicorn.run(app, host=host, port=port, log_level="info")


def start_offline_worker(
    store: LatestSegmentationStore,
    npz_glob: str,
    interval_s: float = 0.1,
) -> threading.Event:
    stop_event = threading.Event()

    def worker() -> None:
        from rock_segmentor import RockSegmentor

        paths = sorted(glob.glob(npz_glob))
        if not paths:
            log.warning("No se encontraron frames offline: %s", npz_glob)
            store.set_source("offline-empty")
            return

        segmentor = RockSegmentor()
        idx = 0
        store.set_source("offline")
        while not stop_event.is_set():
            path = paths[idx % len(paths)]
            data = np.load(path)
            result = segmentor.process(
                data["xyz"].astype(np.float32),
                data.get("confidence"),
                data.get("intensity"),
            )
            store.set_result(result, source="offline", frame=path)
            idx += 1
            stop_event.wait(interval_s)

    threading.Thread(target=worker, daemon=True).start()
    return stop_event
