# 3dcamera - Plataforma ToF + RGB

Repositorio monorepo para captura en edge, procesamiento, API y visualización web.
La idea es separar claramente:
- Edge (picarrocas): captura y subida de datos
- Cloud: API, worker de procesamiento y web para visualización

## Estructura
```
services/
  edge-agent/   # Captura local (ToF + RGB) y subida a storage
  api/          # FastAPI: metadata, índices y acceso a datos
  web/          # FastHTML: UI para visualizar
  worker/       # Jobs async: alineación, pointclouds, thumbnails
packages/
  vision/       # Algoritmos de visión y fusión
  common/       # Schemas, config, logging compartido
infra/
  local/        # docker-compose para dev
  ecs/          # definiciones ECS/Fargate (placeholder)
docker/         # Dockerfiles por servicio
docs/           # arquitectura, guías y contratos
scripts/        # utilidades internas
```

## Flujo (alto nivel)
1. Edge-Agent captura `rgb.png` + `blaze.npz` + metadata.
2. Sube raw a object storage y notifica API.
3. Worker genera derivados (alineación, pointcloud, previews).
4. Web consume API y renderiza en el navegador con Plotly.
