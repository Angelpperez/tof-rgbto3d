from __future__ import annotations

import os

from segmentation_api import LatestSegmentationStore, create_app, start_offline_worker


STORE = LatestSegmentationStore()

if os.getenv("SEG_API_SOURCE", "").strip().lower() == "offline":
    start_offline_worker(
        STORE,
        os.getenv("SEG_API_OFFLINE_GLOB", "frames/*.npz"),
        float(os.getenv("SEG_API_OFFLINE_INTERVAL_S", "0.1")),
    )

app = create_app(
    STORE.get_result,
    get_status=STORE.status,
    title="3dcamera API",
)
