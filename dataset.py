import time, json
from pathlib import Path
import numpy as np
import cv2
from pypylon import pylon

SER_RGB   = "40069848"
SER_BLAZE = "24425138"

N_SAMPLES = 10
SLEEP_S   = 0.25
OUT_DIR = Path("dataset_2cam18-2-2026posatras")
OUT_DIR.mkdir(exist_ok=True)

def open_rgb_by_serial(serial: str):
    tl = pylon.TlFactory.GetInstance()
    for d in tl.EnumerateDevices():
        if d.GetSerialNumber() == serial:
            cam = pylon.InstantCamera(tl.CreateDevice(d))
            cam.Open()
            return cam
    raise RuntimeError(f"No encontré RGB serial={serial}")

def open_blaze_by_serial(serial: str):
    di = pylon.DeviceInfo()
    di.SetDeviceClass("BaslerGTC/Basler/GenTL_Producer_for_Basler_blaze_101_cameras")
    try: di.SetSerialNumber(serial)
    except: pass
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(di))
    cam.Open()
    return cam

def grab_one(cam, timeout_ms=5000):
    cam.StartGrabbingMax(1)
    res = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    if not res.GrabSucceeded():
        raise RuntimeError("Grab falló")
    return res

def set_rgb_color_pixel_format(rgb_cam):
    # Preferimos Bayer8 (lo normal en ace). Si soporta RGB8Packed también sirve.
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
    img = conv.Convert(r).GetArray()  # debe ser HxWx3 uint8
    r.Release()

    # Safety: si algo raro pasó y llega 2D, lo convertimos a 3ch (pero ideal es que NO pase)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def blaze_enable_components(cam):
    cam.ComponentSelector.SetValue("Range")
    cam.ComponentEnable.SetValue(True)
    cam.PixelFormat.SetValue("Coord3D_ABC32f")

    cam.ComponentSelector.SetValue("Intensity")
    cam.ComponentEnable.SetValue(True)
    cam.PixelFormat.SetValue("Mono16")

    cam.ComponentSelector.SetValue("Confidence")
    cam.ComponentEnable.SetValue(True)
    cam.PixelFormat.SetValue("Confidence16")

def grab_blaze_xyz_int_conf(blaze_cam):
    r = grab_one(blaze_cam)
    dc = r.GetDataContainer()

    xyz = intensity = confidence = None

    for i in range(dc.DataComponentCount):
        c = dc.GetDataComponent(i)
        h, w = int(c.Height), int(c.Width)
        arr = np.array(c.Array, copy=True)
        ctype = c.ComponentType

        if ctype == pylon.ComponentType_Range and arr.size == h*w*3:
            xyz = arr.astype(np.float32).reshape(h, w, 3)
        elif ctype == pylon.ComponentType_Intensity and arr.size == h*w:
            intensity = arr.astype(np.uint16).reshape(h, w)
        elif ctype == pylon.ComponentType_Confidence and arr.size == h*w:
            confidence = arr.astype(np.uint16).reshape(h, w)

        c.Release()

    r.Release()

    if xyz is None or intensity is None or confidence is None:
        raise RuntimeError(f"Faltan componentes: xyz={xyz is not None}, intensity={intensity is not None}, confidence={confidence is not None}")

    return xyz, intensity, confidence

def safe(getter, default="N/A"):
    try:
        v = getter()
        return v if v not in ("", None) else default
    except Exception:
        return default

def main():
    rgb_cam = blaze_cam = None
    index_path = OUT_DIR / "index.jsonl"

    try:
        rgb_cam = open_rgb_by_serial(SER_RGB)
        blaze_cam = open_blaze_by_serial(SER_BLAZE)

        # Fuerza color en la RGB ANTES del loop
        set_rgb_color_pixel_format(rgb_cam)

        blaze_enable_components(blaze_cam)

        meta_cam = {
            "rgb": {"model": safe(rgb_cam.GetDeviceInfo().GetModelName), "serial": safe(rgb_cam.GetDeviceInfo().GetSerialNumber)},
            "blaze": {"model": safe(blaze_cam.GetDeviceInfo().GetModelName), "serial": safe(blaze_cam.GetDeviceInfo().GetSerialNumber),
                      "ip": safe(getattr(blaze_cam.GetDeviceInfo(), "GetIpAddress", lambda: "N/A"))}
        }

        with index_path.open("w", encoding="utf-8") as fidx:
            for i in range(N_SAMPLES):
                ts_unix = time.time()
                ts_str = time.strftime("%Y%m%d_%H%M%S")

                rgb = grab_rgb_bgr8(rgb_cam)
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise RuntimeError(f"RGB no es BGR8: shape={rgb.shape} dtype={rgb.dtype}")

                xyz, intensity, confidence = grab_blaze_xyz_int_conf(blaze_cam)

                stem = f"s{i:04d}_{ts_str}"
                rgb_path = OUT_DIR / f"{stem}_rgb.png"
                blaze_path = OUT_DIR / f"{stem}_blaze.npz"

                cv2.imwrite(str(rgb_path), rgb)  # guardará PNG 8-bit 3 canales
                np.savez_compressed(blaze_path, xyz=xyz, intensity=intensity, confidence=confidence)

                record = {
                    "sample_id": i,
                    "timestamp_unix": ts_unix,
                    "timestamp_str": ts_str,
                    "rgb_png": str(rgb_path),
                    "blaze_npz": str(blaze_path),
                    "rgb_shape": list(rgb.shape),
                    "xyz_shape": list(xyz.shape),
                    "intensity_shape": list(intensity.shape),
                    "confidence_shape": list(confidence.shape),
                    "cameras": meta_cam,
                }
                fidx.write(json.dumps(record) + "\n")

                print(f"[OK] {i+1}/{N_SAMPLES} -> {rgb_path.name} + {blaze_path.name}")
                time.sleep(SLEEP_S)

        print("\nDataset listo en:", OUT_DIR.resolve())
        print("Index:", index_path.resolve())

    finally:
        for cam in (rgb_cam, blaze_cam):
            try:
                if cam:
                    try: cam.StopGrabbing()
                    except: pass
                    if cam.IsOpen(): cam.Close()
            except: pass

if __name__ == "__main__":
    main()
