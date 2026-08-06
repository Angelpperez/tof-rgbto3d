# API

FastAPI para:
- Indexar datasets
- Exponer metadata
- Entregar URLs a raw y derivados

Entrypoint: `services/api/app.py`

## Uso local

API incrustada en el visor Open3D:

```powershell
py viewer_seg.py --offline "frames/*.npz" --api
```

Para exponerla en red local o mediante tunel:

```powershell
py viewer_seg.py --offline "frames/*.npz" --api --api-host 0.0.0.0 --api-port 8000
```

Endpoints principales:

- `GET /health`
- `GET /segmentation/latest`
- `GET /segmentation/target`
- `WS /segmentation/stream`
- `GET /viewer`

API standalone con frames offline:

```powershell
$env:SEG_API_SOURCE="offline"
$env:SEG_API_OFFLINE_GLOB="frames/*.npz"
py -m uvicorn services.api.app:app --host 0.0.0.0 --port 8000
```
