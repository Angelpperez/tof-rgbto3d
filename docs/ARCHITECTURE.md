# Arquitectura (alto nivel)

## Objetivo
Sistema para captura ToF+RGB en edge, procesamiento y visualización web.

## Componentes
- Edge-Agent: captura y subida de datos.
- API: catálogo, metadata y acceso a datos.
- Worker: generación de derivados (alineación, pointcloud, previews).
- Web: visualización para usuarios.

## Datos
- Raw: `rgb.png`, `blaze.npz`, `metadata.json`
- Derived: `aligned_rgb.png`, `pointcloud.ply`, `preview.html`
