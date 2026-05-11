# save_frame.py
# Captura N frames de la Blaze y los guarda como .npz para pruebas offline.
# Uso: py save_frame.py --n 5 --out frames/

import argparse, os, time
import numpy as np
from pypylon import pylon

def enable(cam, name, pixfmt):
    cam.ComponentSelector.Value = name
    cam.ComponentEnable.Value   = True
    try:    cam.PixelFormat.Value = pixfmt
    except: pass

def reshape(arr, h, w):
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size == h*w:   return arr.reshape(h, w)
    if arr.ndim == 1 and arr.size == h*w*3: return arr.reshape(h, w, 3)
    return arr

def main(n, out):
    os.makedirs(out, exist_ok=True)
    tl   = pylon.TlFactory.GetInstance()
    devs = tl.EnumerateDevices()
    if not devs: raise SystemExit("No se detecta cámara.")
    cam = pylon.InstantCamera(tl.CreateDevice(devs[0]))
    cam.Open()
    print(f"Cámara: {cam.GetDeviceInfo().GetModelName()} SN:{cam.GetDeviceInfo().GetSerialNumber()}")
    enable(cam, "Intensity",  "Mono16")
    enable(cam, "Range",      "Coord3D_ABC32f")
    enable(cam, "Confidence", "Confidence16")
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    saved = 0
    while saved < n:
        grab = cam.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab.GrabSucceeded(): continue
            dc  = grab.GetDataContainer()
            xyz = intensity = conf = None
            for i in range(dc.DataComponentCount):
                c = dc.GetDataComponent(i)
                try:
                    arr = reshape(c.Array, c.Height, c.Width)
                    if c.ComponentType == pylon.ComponentType_Range:      xyz      = arr
                    elif c.ComponentType == pylon.ComponentType_Intensity: intensity = arr
                    elif c.ComponentType == pylon.ComponentType_Confidence:conf     = arr
                finally:
                    c.Release()
            if xyz is None: continue
            path = os.path.join(out, f"frame_{saved:03d}.npz")
            np.savez_compressed(path, xyz=xyz.astype(np.float32),
                                intensity=intensity, confidence=conf)
            print(f"  Guardado: {path}  xyz={xyz.shape}")
            saved += 1
            time.sleep(0.2)
        finally:
            grab.Release()

    cam.StopGrabbing()
    cam.Close()
    print(f"\nListo — {saved} frames en '{out}/'")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n",   type=int, default=5,        help="Número de frames a capturar")
    p.add_argument("--out", type=str, default="frames", help="Carpeta de salida")
    main(**vars(p.parse_args()))
