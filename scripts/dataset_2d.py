import time, json
from pathlib import Path
import numpy as np
import cv2
from pypylon import pylon

# ====== CONFIG ======
SER_RGB = "40069848"

N_SAMPLES = 10
SLEEP_S   = 0.25

OUT_DIR = Path("dataset_rgb_only_18-2-2026posatras")
OUT_DIR.mkdir(exist_ok=True)
# ====================

def open_rgb_by_serial(serial: str):
    tl = pylon.TlFactory.GetInstance()
    for d in tl.EnumerateDevices():
        if d.GetSerialNumber() == serial:
            cam = pylon.InstantCamera(tl.CreateDevice(d))
            cam.Open()
            return cam
    raise RuntimeError(f"No encontré RGB serial={serial}")

def grab_one(cam, timeout_ms=5000):
    cam.StartGrabbingMax(1)
    res = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    if not res.GrabSucceeded():
        raise RuntimeError("Grab falló")
    return res

def set_rgb_color_pixel_format(rgb_cam):
    # Intentamos forzar un formato "raw" Bayer o directamente RGB/BGR packed.
    candidates = ["BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8", "RGB8Packed", "BGR8Packed"]
    for fmt in candidates:
        try:
            rgb_cam.PixelFormat.SetValue(fmt)
            print(f"[RGB] PixelFormat -> {fmt}")
            return fmt
        except Exception:
            pass
    try:
        print("[RGB] No pude setear formato color. Disponibles:", rgb_cam.PixelFormat.GetSymbolics())
    except Exception:
        print("[RGB] No pude setear formato color y no pude listar symbolics.")
    return None

def grab_rgb_bgr8(rgb_cam):
    # Converter SIEMPRE a BGR8 (3 canales, 8-bit)
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_BGR8packed
    conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    r = grab_one(rgb_cam)
    img = conv.Convert(r).GetArray()  # HxWx3 uint8 (ideal)
    r.Release()

    # Safety: si llega 2D, lo convertimos a 3ch
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise RuntimeError(f"RGB no es BGR8: shape={img.shape} dtype={img.dtype}")

    return img

def safe(getter, default="N/A"):
    try:
        v = getter()
        return v if v not in ("", None) else default
    except Exception:
        return default

def main():
    rgb_cam = None
    index_path = OUT_DIR / "index.jsonl"

    try:
        rgb_cam = open_rgb_by_serial(SER_RGB)

        # Fuerza color antes del loop
        set_rgb_color_pixel_format(rgb_cam)

        meta_cam = {
            "rgb": {
                "model":  safe(rgb_cam.GetDeviceInfo().GetModelName),
                "serial": safe(rgb_cam.GetDeviceInfo().GetSerialNumber),
            }
        }

        with index_path.open("w", encoding="utf-8") as fidx:
            for i in range(N_SAMPLES):
                ts_unix = time.time()
                ts_str  = time.strftime("%Y%m%d_%H%M%S")

                rgb = grab_rgb_bgr8(rgb_cam)

                stem = f"s{i:04d}_{ts_str}"
                rgb_path = OUT_DIR / f"{stem}_rgb.png"

                cv2.imwrite(str(rgb_path), rgb)  # PNG 8-bit 3 canales

                record = {
                    "sample_id": i,
                    "timestamp_unix": ts_unix,
                    "timestamp_str": ts_str,
                    "rgb_png": str(rgb_path),
                    "rgb_shape": list(rgb.shape),
                    "cameras": meta_cam,
                }
                fidx.write(json.dumps(record) + "\n")

                print(f"[OK] {i+1}/{N_SAMPLES} -> {rgb_path.name}")
                time.sleep(SLEEP_S)

        print("\nDataset listo en:", OUT_DIR.resolve())
        print("Index:", index_path.resolve())

    finally:
        try:
            if rgb_cam:
                try:
                    rgb_cam.StopGrabbing()
                except Exception:
                    pass
                if rgb_cam.IsOpen():
                    rgb_cam.Close()
        except Exception:
            pass

if __name__ == "__main__":
    main()