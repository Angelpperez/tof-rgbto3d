from __future__ import annotations

import logging
import os
import socket
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("simulink_udp")


def load_env_file(path: str = ".env.simulink") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "s"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        log.warning("%s invalido=%r; usando %d", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        log.warning("%s invalido=%r; usando %.3f", name, value, default)
        return default


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class SimulinkUdpSender:
    """Envia [x,y,z] en metros a Simulink como 3 doubles little-endian."""

    def __init__(
        self,
        enabled: bool,
        dest_ip: str,
        dest_port: int,
        source_ip: str,
        source_port: int,
        data_type: str,
        byte_order: str,
        send_on_hold: bool,
        min_period_s: float,
    ) -> None:
        self.enabled = enabled
        self.dest = (dest_ip, dest_port)
        self.source = (source_ip, source_port)
        self.send_on_hold = send_on_hold
        self.min_period_s = max(0.0, min_period_s)
        self._last_send = 0.0
        self._sock: Optional[socket.socket] = None
        self._fmt = self._make_struct_format(data_type, byte_order)

        if not self.enabled:
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind(self.source)
        except OSError as exc:
            log.warning(
                "UDP Simulink deshabilitado: no pude usar %s:%d (%s)",
                self.source[0],
                self.source[1],
                exc,
            )
            self.enabled = False
            self._sock.close()
            self._sock = None
            return

        log.info(
            "UDP Simulink activo: %s:%d -> %s:%d formato=%s",
            self.source[0],
            self.source[1],
            self.dest[0],
            self.dest[1],
            self._fmt,
        )

    @classmethod
    def from_env(cls) -> "SimulinkUdpSender":
        load_env_file()
        return cls(
            enabled=_env_bool("SIMULINK_UDP_ENABLED", False),
            dest_ip=_env_first(
                ("SIMULINK_UDP_DEST_IP", "SIMULINK_UDP_REMOTE_ADDRESS"),
                "127.0.0.1",
            ),
            dest_port=_env_int(
                "SIMULINK_UDP_DEST_PORT",
                _env_int("SIMULINK_UDP_REMOTE_PORT", 55001),
            ),
            source_ip=_env_first(
                ("SIMULINK_UDP_SOURCE_IP", "SIMULINK_UDP_LOCAL_ADDRESS"),
                "127.0.0.1",
            ),
            source_port=_env_int(
                "SIMULINK_UDP_SOURCE_PORT",
                _env_int("SIMULINK_UDP_LOCAL_PORT", 55000),
            ),
            data_type=os.getenv("SIMULINK_UDP_DATA_TYPE", "double"),
            byte_order=os.getenv("SIMULINK_UDP_BYTE_ORDER", "little"),
            send_on_hold=_env_bool("SIMULINK_UDP_SEND_ON_HOLD", True),
            min_period_s=_env_float("SIMULINK_UDP_MIN_PERIOD_S", 0.0),
        )

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send_xyz(self, xyz_m: np.ndarray, tracking_status: str = "") -> bool:
        if not self.enabled or self._sock is None:
            return False
        if tracking_status == "hold" and not self.send_on_hold:
            return False

        now = time.perf_counter()
        if self.min_period_s > 0.0 and now - self._last_send < self.min_period_s:
            return False

        xyz = np.asarray(xyz_m, dtype=np.float64).reshape(3)
        payload = struct.pack(self._fmt, float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self._sock.sendto(payload, self.dest)
        self._last_send = now
        return True

    @staticmethod
    def _make_struct_format(data_type: str, byte_order: str) -> str:
        order = "<" if byte_order.strip().lower() in {"little", "little-endian", "le"} else ">"
        dtype = data_type.strip().lower()
        if dtype in {"double", "float64"}:
            return order + "3d"
        if dtype in {"single", "float32"}:
            return order + "3f"
        raise ValueError(f"SIMULINK_UDP_DATA_TYPE no soportado: {data_type!r}")
