from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

import numpy as np
import open3d as o3d
from pypylon import pylon, genicam

# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("tof_rgb")

# ---------------------------
# CONFIGS
# ---------------------------
@dataclass(frozen=True)
class BlazeConfig:
    operating_mode: str = "Long Range"   # como en pylon Viewer
    fast_mode: bool = False
    exposure_us: float = 740.0           # microsegundos
    min_depth_mm: int = 0
    max_depth_mm: int = 9990

    # Componentes a activar
    enable_range_xyz: bool = True
    enable_intensity: bool = True
    enable_confidence: bool = True

    # Pixel formats deseados
    pf_xyz: str = "Coord3D_ABC32f"
    pf_intensity: str = "Mono16"
    pf_confidence: str = "Confidence16"


@dataclass(frozen=True)
class FusionConfig:
    ser_rgb: str
    ser_blaze: str
    xyz_scale: float = 0.001  # blaze suele venir en mm -> m; si ya viene en m usa 1.0

    # PLACEHOLDERS: calib real después
    fx: float = 900.0
    fy: float = 900.0


# ---------------------------
# GenICam helpers (robustos)
# ---------------------------
class NodeUtils:
    @staticmethod
    def _get_node(nm: Any, name: str):
        try:
            node = nm.GetNode(name)
            if node is None:
                return None
            if not genicam.IsAvailable(node):
                return None
            return node
        except Exception:
            return None

    @staticmethod
    def set_enum(nm: Any, names: List[str], value: str, fuzzy: bool = True) -> bool:
        """
        Intenta setear una enumeración probando varios nombres de nodo.
        fuzzy=True intenta variantes del string (sin espacios, underscores).
        """
        candidates = [value]
        if fuzzy:
            candidates += [
                value.replace(" ", ""),
                value.replace(" ", "_"),
                value.replace("_", " "),
            ]

        for name in names:
            node = NodeUtils._get_node(nm, name)
            if node is None:
                continue
            try:
                enum_ptr = genicam.CEnumerationPtr(node)
                if not genicam.IsWritable(enum_ptr):
                    log.warning("Enum '%s' existe pero no es writable ahora", name)
                    continue

                # Probar candidatos
                last_err = None
                for v in candidates:
                    try:
                        enum_ptr.FromString(v)
                        log.info("Set %s = %s", name, v)
                        return True
                    except Exception as e:
                        last_err = e

                # Si falla, loggear valores disponibles
                try:
                    avail = []
                    for i in range(enum_ptr.GetEntries().size()):
                        e = enum_ptr.GetEntries().at(i)
                        if genicam.IsAvailable(e):
                            avail.append(e.GetSymbolic())
                    log.error("No pude setear %s. Quise '%s'. Disponibles: %s", name, value, avail)
                except Exception:
                    log.error("No pude setear %s. Quise '%s'. Error: %r", name, value, last_err)
            except Exception as e:
                log.warning("Falló set_enum en nodo %s: %r", name, e)
        return False

    @staticmethod
    def set_float(nm: Any, names: List[str], value: float) -> bool:
        for name in names:
            node = NodeUtils._get_node(nm, name)
            if node is None:
                continue
            try:
                f = genicam.CFloatPtr(node)
                if not genicam.IsWritable(f):
                    log.warning("Float '%s' existe pero no es writable ahora", name)
                    continue
                f.SetValue(float(value))
                log.info("Set %s = %s", name, value)
                return True
            except Exception as e:
                log.warning("Falló set_float en %s: %r", name, e)
        return False

    @staticmethod
    def set_int(nm: Any, names: List[str], value: int) -> bool:
        for name in names:
            node = NodeUtils._get_node(nm, name)
            if node is None:
                continue
            try:
                it = genicam.CIntegerPtr(node)
                if not genicam.IsWritable(it):
                    log.warning("Int '%s' existe pero no es writable ahora", name)
                    continue
                it.SetValue(int(value))
                log.info("Set %s = %s", name, value)
                return True
            except Exception as e:
                log.warning("Falló set_int en %s: %r", name, e)
        return False

    @staticmethod
    def set_bool(nm: Any, names: List[str], value: bool) -> bool:
        for name in names:
            node = NodeUtils._get_node(nm, name)
            if node is None:
                continue
            try:
                b = genicam.CBooleanPtr(node)
                if not genicam.IsWritable(b):
                    log.warning("Bool '%s' existe pero no es writable ahora", name)
                    continue
                b.SetValue(bool(value))
                log.info("Set %s = %s", name, value)
                return True
            except Exception as e:
                log.warning("Falló set_bool en %s: %r", name, e)
        return False


# ---------------------------
# Camera wrappers
# ---------------------------
class BaslerRGBCamera:
    def __init__(self, serial: str):
        self.serial = serial
        self.cam: Optional[pylon.InstantCamera] = None
        self._conv = pylon.ImageFormatConverter()
        self._conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def open(self) -> None:
        tl = pylon.TlFactory.GetInstance()
        for d in tl.EnumerateDevices():
            if d.GetSerialNumber() == self.serial:
                self.cam = pylon.InstantCamera(tl.CreateDevice(d))
                self.cam.Open()
                di = self.cam.GetDeviceInfo()
                log.info("RGB abierto: %s SN=%s", di.GetModelName(), di.GetSerialNumber())
                return
        raise RuntimeError(f"No encontré RGB serial={self.serial}")

    def grab_bgr8(self, timeout_ms: int = 5000) -> np.ndarray:
        assert self.cam is not None
        self.cam.StartGrabbingMax(1)
        r = self.cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        if not r.GrabSucceeded():
            raise RuntimeError("RGB grab falló")
        img = self._conv.Convert(r).GetArray()
        r.Release()
        return img

    def close(self) -> None:
        if self.cam:
            try:
                self.cam.StopGrabbing()
            except Exception:
                pass
            try:
                if self.cam.IsOpen():
                    self.cam.Close()
            except Exception:
                pass
            self.cam = None


class BaslerBlazeToF:
    """
    Wrapper blaze usando DataContainer (ComponentSelector).
    """
    DEVICE_CLASS = "BaslerGTC/Basler/GenTL_Producer_for_Basler_blaze_101_cameras"

    def __init__(self, serial: str, cfg: BlazeConfig):
        self.serial = serial
        self.cfg = cfg
        self.cam: Optional[pylon.InstantCamera] = None

    def open(self) -> None:
        # Abrir por DeviceClass (blaze)
        di = pylon.DeviceInfo()
        di.SetDeviceClass(self.DEVICE_CLASS)
        try:
            di.SetSerialNumber(self.serial)
        except Exception:
            pass

        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(di))
        self.cam.Open()

        # Validación “best effort”
        try:
            got = self.cam.GetDeviceInfo().GetSerialNumber()
            log.info("Blaze abierto: %s SN=%s", self.cam.GetDeviceInfo().GetModelName(), got)
            if got != self.serial:
                log.warning("Abrí SN=%s pero querías SN=%s", got, self.serial)
        except Exception:
            log.info("Blaze abierto (no pude validar serial)")

    def configure(self) -> None:
        """
        Setea OperatingMode / FastMode / ExposureTime / MinDepth / MaxDepth + componentes.
        IMPORTANTE: llamar antes de grab.
        """
        assert self.cam is not None
        nm = self.cam.GetNodeMap()

        # 1) Operating mode (el nombre exacto puede variar por firmware)
        NodeUtils.set_enum(nm, ["OperatingMode", "ToFOperatingMode", "Scan3dOperatingMode"], self.cfg.operating_mode)

        # 2) Fast mode
        NodeUtils.set_bool(nm, ["FastMode", "ToFFastMode"], self.cfg.fast_mode)

        # 3) Exposure time (us)
        NodeUtils.set_float(nm, ["ExposureTime", "ExposureTimeAbs"], self.cfg.exposure_us)

        # 4) Min/Max depth (mm) (nombres pueden variar)
        NodeUtils.set_int(nm, ["MinimumDepth", "MinDepth", "DepthMin", "MinDistance"], self.cfg.min_depth_mm)
        NodeUtils.set_int(nm, ["MaximumDepth", "MaxDepth", "DepthMax", "MaxDistance"], self.cfg.max_depth_mm)

        # 5) Componentes + PixelFormat por componente (como ya venías haciendo)
        self._enable_components()

    def _set_component_pf(self, comp: str, pixfmt: str) -> None:
        assert self.cam is not None
        self.cam.ComponentSelector.SetValue(comp)
        self.cam.ComponentEnable.SetValue(True)
        try:
            node_pf = self.cam.GetNodeMap().GetNode("PixelFormat")
            if node_pf is not None and genicam.IsWritable(node_pf):
                self.cam.PixelFormat.SetValue(pixfmt)
                log.info("Component %s PixelFormat=%s", comp, pixfmt)
        except Exception as e:
            log.warning("No pude setear PixelFormat=%s en %s: %r", pixfmt, comp, e)

    def _enable_components(self) -> None:
        if self.cfg.enable_range_xyz:
            self._set_component_pf("Range", self.cfg.pf_xyz)
        if self.cfg.enable_intensity:
            self._set_component_pf("Intensity", self.cfg.pf_intensity)
        if self.cfg.enable_confidence:
            self._set_component_pf("Confidence", self.cfg.pf_confidence)

    def grab_components(self, timeout_ms: int = 5000) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retorna: xyz(H,W,3) float32, intensity(H,W) uint16|None, confidence(H,W) uint16|None
        """
        assert self.cam is not None

        self.cam.StartGrabbingMax(1)
        r = self.cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        if not r.GrabSucceeded():
            raise RuntimeError("Blaze grab falló")

        dc = r.GetDataContainer()

        xyz = None
        intensity = None
        confidence = None

        for i in range(dc.DataComponentCount):
            c = dc.GetDataComponent(i)
            h, w = int(c.Height), int(c.Width)
            arr = np.array(c.Array, copy=True)  # copiar antes de Release
            ctype = c.ComponentType

            if ctype == pylon.ComponentType_Range:
                if arr.size == h * w * 3:
                    xyz = arr.astype(np.float32).reshape(h, w, 3)
            elif ctype == pylon.ComponentType_Intensity and arr.size == h * w:
                intensity = arr.astype(np.uint16).reshape(h, w)
            elif ctype == pylon.ComponentType_Confidence and arr.size == h * w:
                confidence = arr.astype(np.uint16).reshape(h, w)

            c.Release()

        r.Release()

        if xyz is None:
            raise RuntimeError("No obtuve XYZ (Range H*W*3). Revisa que Range esté en Coord3D_ABC32f.")
        return xyz, intensity, confidence

    def close(self) -> None:
        if self.cam:
            try:
                self.cam.StopGrabbing()
            except Exception:
                pass
            try:
                if self.cam.IsOpen():
                    self.cam.Close()
            except Exception:
                pass
            self.cam = None


# ---------------------------
# Fusion + Viewer
# ---------------------------
class FusionOneFrame:
    def __init__(self, cfg: FusionConfig, blaze_cfg: BlazeConfig):
        self.cfg = cfg
        self.rgb = BaslerRGBCamera(cfg.ser_rgb)
        self.blaze = BaslerBlazeToF(cfg.ser_blaze, blaze_cfg)

    def run(self) -> None:
        try:
            self.rgb.open()
            self.blaze.open()
            self.blaze.configure()  # <-- aquí controlas exposure/operating mode/etc

            rgb_bgr = self.rgb.grab_bgr8()
            xyz, intensity, confidence = self.blaze.grab_components()

            log.info("RGB: %s %s", rgb_bgr.shape, rgb_bgr.dtype)
            log.info("XYZ: %s %s", xyz.shape, xyz.dtype)
            if confidence is not None:
                log.info("CONF: %s %s", confidence.shape, confidence.dtype)

            pts, cols = self._colorize(xyz, rgb_bgr, confidence=confidence, conf_min=1)
            log.info("Puntos colorizados: %d", len(pts))

            self._show_o3d(pts, cols)
        finally:
            self.rgb.close()
            self.blaze.close()

    def _colorize(self, xyz_hw3: np.ndarray, rgb_bgr: np.ndarray,
                  confidence: Optional[np.ndarray] = None, conf_min: int = 1):
        h, w = rgb_bgr.shape[:2]
        fx, fy = self.cfg.fx, self.cfg.fy
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)

        # PLACEHOLDERS (calibración real después)
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3, 1), dtype=np.float32)

        pts = xyz_hw3.reshape(-1, 3).astype(np.float32) * float(self.cfg.xyz_scale)
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        if confidence is not None:
            valid &= (confidence.reshape(-1) >= conf_min)

        pts = pts[valid]
        pts_rgb = pts @ R.T + t.reshape(1, 3)

        x, y, z = pts_rgb[:, 0], pts_rgb[:, 1], pts_rgb[:, 2]
        u = (K[0, 0] * (x / z) + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * (y / z) + K[1, 2]).astype(np.int32)

        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        pts_rgb = pts_rgb[in_img]
        u, v = u[in_img], v[in_img]

        colors_bgr = rgb_bgr[v, u].astype(np.float32) / 255.0
        colors_rgb = colors_bgr[:, ::-1]
        return pts_rgb, colors_rgb

    @staticmethod
    def _show_o3d(points_xyz: np.ndarray, colors_rgb: np.ndarray) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
        o3d.visualization.draw_geometries([pcd])


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    blaze_cfg = BlazeConfig(
        operating_mode="Long Range",
        fast_mode=False,
        exposure_us=740.0,      # <-- CAMBIA AQUÍ
        min_depth_mm=0,
        max_depth_mm=9990
    )

    fusion_cfg = FusionConfig(
        ser_rgb="40069848",
        ser_blaze="24425138",
        xyz_scale=0.001,
        fx=900.0,
        fy=900.0
    )

    FusionOneFrame(fusion_cfg, blaze_cfg).run()
