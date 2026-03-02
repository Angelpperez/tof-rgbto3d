import time, json
from pathlib import Path
import numpy as np
from pypylon import pylon

# ====== CONFIG ======
SER_BLAZE = "24425138"

N_SAMPLES = 10
SLEEP_S   = 0.25

OUT_DIR = Path("dataset_blaze_only_27-2-2026posatras4")
OUT_DIR.mkdir(exist_ok=True)
# ====================

def open_blaze_by_serial(serial: str):
    di = pylon.DeviceInfo()
    di.SetDeviceClass("BaslerGTC/Basler/GenTL_Producer_for_Basler_blaze_101_cameras")
    try:
        di.SetSerialNumber(serial)
    except Exception:
        pass
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(di))
    cam.Open()
    return cam

def grab_one(cam, timeout_ms=5000):
    cam.StartGrabbingMax(1)
    res = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    if not res.GrabSucceeded():
        raise RuntimeError("Grab falló")
    return res

def safe(getter, default="N/A"):
    try:
        v = getter()
        return v if v not in ("", None) else default
    except Exception:
        return default

def blaze_enable_components(cam):
    # Range -> Coord3D_ABC32f (X,Y,Z float32)
    cam.ComponentSelector.SetValue("Range")
    cam.ComponentEnable.SetValue(True)
    cam.PixelFormat.SetValue("Coord3D_ABC32f")

    # Intensity -> Mono16
    cam.ComponentSelector.SetValue("Intensity")
    cam.ComponentEnable.SetValue(True)
    cam.PixelFormat.SetValue("Mono16")

    # Confidence -> Confidence16
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

        if ctype == pylon.ComponentType_Range and arr.size == h * w * 3:
            xyz = arr.astype(np.float32).reshape(h, w, 3)
        elif ctype == pylon.ComponentType_Intensity and arr.size == h * w:
            intensity = arr.astype(np.uint16).reshape(h, w)
        elif ctype == pylon.ComponentType_Confidence and arr.size == h * w:
            confidence = arr.astype(np.uint16).reshape(h, w)

        c.Release()

    r.Release()

    if xyz is None or intensity is None or confidence is None:
        raise RuntimeError(
            f"Faltan componentes: xyz={xyz is not None}, intensity={intensity is not None}, confidence={confidence is not None}"
        )

    return xyz, intensity, confidence

def main():
    blaze_cam = None
    index_path = OUT_DIR / "index.jsonl"

    try:
        blaze_cam = open_blaze_by_serial(SER_BLAZE)
        blaze_enable_components(blaze_cam)

        meta_cam = {
            "blaze": {
                "model":  safe(blaze_cam.GetDeviceInfo().GetModelName),
                "serial": safe(blaze_cam.GetDeviceInfo().GetSerialNumber),
                "ip":     safe(getattr(blaze_cam.GetDeviceInfo(), "GetIpAddress", lambda: "N/A")),
            }
        }

        with index_path.open("w", encoding="utf-8") as fidx:
            for i in range(N_SAMPLES):
                ts_unix = time.time()
                ts_str  = time.strftime("%Y%m%d_%H%M%S")

                xyz, intensity, confidence = grab_blaze_xyz_int_conf(blaze_cam)

                stem = f"s{i:04d}_{ts_str}"
                blaze_path = OUT_DIR / f"{stem}_blaze.npz"

                np.savez_compressed(
                    blaze_path,
                    xyz=xyz,
                    intensity=intensity,
                    confidence=confidence
                )

                record = {
                    "sample_id": i,
                    "timestamp_unix": ts_unix,
                    "timestamp_str": ts_str,
                    "blaze_npz": str(blaze_path),
                    "xyz_shape": list(xyz.shape),
                    "intensity_shape": list(intensity.shape),
                    "confidence_shape": list(confidence.shape),
                    "cameras": meta_cam,
                }
                fidx.write(json.dumps(record) + "\n")

                print(f"[OK] {i+1}/{N_SAMPLES} -> {blaze_path.name}")
                time.sleep(SLEEP_S)

        print("\nDataset listo en:", OUT_DIR.resolve())
        print("Index:", index_path.resolve())

    finally:
        try:
            if blaze_cam:
                try:
                    blaze_cam.StopGrabbing()
                except Exception:
                    pass
                if blaze_cam.IsOpen():
                    blaze_cam.Close()
        except Exception:
            pass

if __name__ == "__main__":
    main()