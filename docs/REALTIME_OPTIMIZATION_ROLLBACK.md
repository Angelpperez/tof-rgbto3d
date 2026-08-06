# Plan de optimizacion realtime y rollback

Fecha base: 2026-08-06 11:31:34 -04:00

Commit base: `b9159abcec8d01f93dc817e86d8b8984e02be71b`

Importante: el repo no estaba limpio al crear este documento. El punto anterior que queremos preservar es el working tree actual, no solo el commit base.

## Estado actual observado

Archivos modificados:

- `docker/Dockerfile.api`
- `requirements.txt`
- `rock_segmentor.py`
- `server_blaze_seg.py`
- `services/api/README.md`
- `services/api/app.py`
- `viewer_seg.py`

Archivos no trackeados:

- `Microsoft/`
- `segmentation_api.py`

## Objetivo de la modificacion

Hacer que la aplicacion se sienta en tiempo real sin romper la segmentacion existente:

- Captura: seguir leyendo siempre el frame mas reciente.
- Render: actualizar la nube cruda submuestreada a 15-30 fps.
- Segmentacion: correr asincronicamente sobre el ultimo frame disponible, sin cola.
- Overlay: mantener el ultimo OBB/resultado de segmentacion hasta que termine uno nuevo.

## Problema actual

`SEG_EVERY = 5` existe, pero el contador actual no esta amarrado a frames reales nuevos. El hilo de segmentacion puede incrementar el contador por vueltas del loop y procesar repetidamente el mismo frame.

Ademas, despues de la primera segmentacion, el visor mantiene una nube segmentada cacheada. Eso hace que la escena visible quede atada a la velocidad de segmentacion, no a la velocidad de captura.

## Cambio propuesto por fases

### Fase 1 - Control de frames reales

Archivos esperados:

- `viewer_seg.py`
- posiblemente `server_blaze_seg.py`

Cambios:

- Agregar `LATEST["frame_id"]`.
- Incrementar `frame_id` una vez por captura exitosa.
- Guardar `frame_id` junto con `xyz`, `conf` e `intens`.
- Hacer que `seg_loop` solo procese cuando `frame_id` sea nuevo.
- Aplicar `SEG_EVERY` como diferencia real de frames:

```python
if frame_id - last_processed_frame_id < SEG_EVERY:
    time.sleep(0.005)
    continue
```

### Fase 2 - Render vivo desacoplado

Archivos esperados:

- `viewer_seg.py`

Cambios:

- Mostrar nube cruda submuestreada en cada frame nuevo.
- Mantener ultimo OBB/labels de segmentacion como overlay.
- Actualizar geometria solo cuando cambie `frame_id` o cuando cambie `seg.timestamp`.
- Evitar reconstruir la nube cacheada si no hubo cambios.

Parametros iniciales sugeridos:

- `RENDER_STRIDE = 4`
- `SEG_STRIDE = 3`
- `SEG_EVERY = 5`

### Fase 3 - Reduccion de costo de segmentacion

Archivos esperados:

- `rock_segmentor.py`
- `viewer_seg.py`
- posiblemente `server_blaze_seg.py`

Cambios:

- Pasar a `RockSegmentor.process()` una version submuestreada para segmentacion.
- Mantener coordenadas en la misma escala para que OBB y top point sigan consistentes.
- Reducir `max_object_pts` a un rango inicial de 15000-30000 si la precision se mantiene.

## Criterios de exito

- El visor no se congela esperando DBSCAN.
- La nube cruda se mueve fluida aunque el OBB tarde mas.
- El OBB anterior permanece visible hasta que llega el nuevo.
- No se acumula backlog de frames.
- Los logs muestran que no se procesa el mismo `frame_id` repetidamente.

## Criterios de rollback

Volver al estado anterior si ocurre cualquiera de estos casos:

- El visor deja de abrir.
- La camara deja de capturar frames.
- El OBB aparece en coordenadas incorrectas.
- El overlay queda desfasado de forma inutil para operar.
- La segmentacion pierde el objetivo de forma recurrente.
- El cambio afecta rutas no relacionadas como API, Docker o servicios.

## Procedimiento recomendado antes de tocar codigo

Como el repo ya tiene cambios sin commit, no usar `git reset --hard` como primer recurso.

Crear un snapshot local del estado actual antes de modificar:

```powershell
New-Item -ItemType Directory -Force .rollback
git diff > .rollback\before_realtime_optimization.patch
git status --short > .rollback\before_realtime_optimization.status.txt
```

Para archivos no trackeados importantes, copiarlos manualmente a `.rollback` o confirmarlos antes de empezar. En este estado hay al menos:

- `segmentation_api.py`
- `Microsoft/`

## Procedimiento de rollback seguro

Si las modificaciones nuevas fallan, revertir solo los archivos tocados por esta optimizacion.

Opcion preferida:

```powershell
git diff > .rollback\failed_realtime_optimization.patch
git apply -R .rollback\failed_realtime_optimization.patch
```

Si se guardo el snapshot inicial y se necesita volver exactamente al punto anterior:

```powershell
git apply -R .rollback\failed_realtime_optimization.patch
git apply .rollback\before_realtime_optimization.patch
```

Si hay conflictos, revisar archivo por archivo. No ejecutar `git reset --hard` porque perderia cambios previos no relacionados.

## Validacion minima despues del cambio

Sin camara:

```powershell
.\.venv3d\Scripts\python.exe -m py_compile viewer_seg.py rock_segmentor.py server_blaze_seg.py
.\.venv3d\Scripts\python.exe viewer_seg.py --offline "frames/*.npz"
```

Con camara:

```powershell
.\.venv3d\Scripts\python.exe viewer_seg.py
```

Metricas a mirar:

- `frame_id` capturado aumenta de forma continua.
- `seg_frame_id` salta cada `SEG_EVERY` frames reales.
- `result.elapsed_ms` baja al usar `SEG_STRIDE`.
- El render sigue respondiendo mientras `result.elapsed_ms` es alto.

