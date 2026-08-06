bl_info = {
    "name": "Blaze Realtime PointCloud Viewer",
    "author": "Angel + ChatGPT",
    "version": (0, 4, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Blaze",
    "category": "3D View",
}

import bpy
import threading
import socket
import struct
import time
import zlib
from mathutils import Vector

try:
    import numpy as np
except Exception:
    np = None

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
UDP_V2_PORT = 5007

TCP_IP = "127.0.0.1"
TCP_PORT = 6006

V2_MAGIC = b"BLZ2"
V2_VERSION = 2
V2_HEADER = struct.Struct("<4sHHQQHHIII")

STATE = {
    "running": False,
    "stream_mode": "legacy",
    "points": [],
    "colors": None,
    "latest_frame": None,
    "displayed_frame_id": 0,
    "last_frame_id": 0,
    "last_latency_ms": 0.0,
    "dropped_frames": 0,
    "obbs": [],
    "lock": threading.Lock(),
    "obj_name": "BlazePointCloud",
    "bbox_name": "BlazeBBox",
    "last_n": 0,

    # Auto-fit settings
    "auto_fit": True,
    "target_size": 2.0,     # tamaño objetivo (en unidades Blender) para el mayor eje
    "point_radius": 0.01,

    # axis tweaks (blaze->blender)
    "flip_x": False,
    "flip_y": False,
    "flip_z": False,
    "swap_yz": False,       # si se ve “acostado”, prueba True
}


def ensure_pointcloud_object():
    name = STATE["obj_name"]
    if name in bpy.data.objects:
        return bpy.data.objects[name]

    pc = bpy.data.pointclouds.new(name + "Data")
    obj = bpy.data.objects.new(name, pc)
    bpy.context.collection.objects.link(obj)
    obj.show_in_front = True
    return obj


def ensure_bbox_object():
    name = STATE["bbox_name"]
    if name in bpy.data.objects:
        return bpy.data.objects[name]

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.display_type = 'WIRE'
    obj.show_in_front = True
    obj.color = (0.1, 0.3, 1.0, 1.0)
    return obj


def apply_axis_ops(p):
    x, y, z = p
    if STATE["swap_yz"]:
        y, z = z, y
    if STATE["flip_x"]:
        x = -x
    if STATE["flip_y"]:
        y = -y
    if STATE["flip_z"]:
        z = -z
    return (x, y, z)


def apply_axis_ops_array(pts):
    if np is None:
        return [apply_axis_ops(p) for p in pts]

    out = np.asarray(pts, dtype=np.float32).copy()
    if STATE["swap_yz"]:
        out[:, [1, 2]] = out[:, [2, 1]]
    if STATE["flip_x"]:
        out[:, 0] *= -1.0
    if STATE["flip_y"]:
        out[:, 1] *= -1.0
    if STATE["flip_z"]:
        out[:, 2] *= -1.0
    return out


def center_and_scale(pts):
    # pts: list[(x,y,z)]
    # centrar
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    cz = (min(zs) + max(zs)) * 0.5

    # tamaño (extent)
    ex = max(xs) - min(xs)
    ey = max(ys) - min(ys)
    ez = max(zs) - min(zs)
    max_extent = max(ex, ey, ez)

    if max_extent < 1e-9:
        s = 1.0
    else:
        s = STATE["target_size"] / max_extent

    out = []
    for (x, y, z) in pts:
        out.append(((x - cx) * s, (y - cy) * s, (z - cz) * s))
    return out


def center_and_scale_frame(pts, obbs):
    if np is None:
        fit_pts = center_and_scale(pts)
        fit_obbs = [center_and_scale(corners) for corners in obbs]
        return fit_pts, fit_obbs

    if pts.shape[0] == 0:
        return pts, obbs

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) * 0.5
    max_extent = float(np.max(mx - mn))
    scale = 1.0 if max_extent < 1e-9 else STATE["target_size"] / max_extent

    fit_pts = (pts - center.reshape(1, 3)) * scale
    fit_obbs = []
    for corners in obbs:
        fit_obbs.append((corners - center.reshape(1, 3)) * scale)
    return fit_pts.astype(np.float32, copy=False), fit_obbs


def set_pointcloud_data(obj, pts, radius=0.01):
    pc = obj.data
    n = len(pts)
    pc.points.clear()
    pc.points.add(n)

    try:
        if np is not None:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            pc.points.foreach_set("co", arr.reshape(-1))
            pc.points.foreach_set("radius", np.full(n, radius, dtype=np.float32))
        else:
            raise RuntimeError("numpy unavailable")
    except Exception:
        for i, p in enumerate(pts):
            pc.points[i].co = p
            pc.points[i].radius = radius

    pc.update()


def set_bbox_from_corners(obj, corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))

    v = [
        (mn[0], mn[1], mn[2]),
        (mx[0], mn[1], mn[2]),
        (mx[0], mx[1], mn[2]),
        (mn[0], mx[1], mn[2]),
        (mn[0], mn[1], mx[2]),
        (mx[0], mn[1], mx[2]),
        (mx[0], mx[1], mx[2]),
        (mn[0], mx[1], mx[2]),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(v, edges, [])
    mesh.update()


def set_obb_from_corners(obj, corners):
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    verts = corners.tolist() if np is not None and hasattr(corners, "tolist") else list(corners)
    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(verts, edges, [])
    mesh.update()


def parse_obb_payload(data, offset):
    if offset + 2 > len(data):
        return [], offset

    count = struct.unpack_from("<H", data, offset)[0]
    offset += 2
    obbs = []

    for _ in range(count):
        if offset + 6 > len(data):
            break
        prob = struct.unpack_from("<f", data, offset)[0]
        offset += 4
        corner_count = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        corners_len = corner_count * 12
        if offset + corners_len > len(data):
            break

        if np is not None:
            corners = np.frombuffer(
                data,
                dtype="<f4",
                count=corner_count * 3,
                offset=offset,
            ).reshape(corner_count, 3).copy()
        else:
            corners = []
            for i in range(corner_count):
                corners.append(struct.unpack_from("<fff", data, offset + i * 12))
        offset += corners_len
        obbs.append({"prob": prob, "corners": corners})

    return obbs, offset


def decode_v2_frame(payload, frame_id, timestamp_ns):
    if len(payload) < 4:
        return None

    n = struct.unpack_from("<I", payload, 0)[0]
    xyz_offset = 4
    xyz_len = n * 12
    rgb_offset = xyz_offset + xyz_len
    rgb_len = n * 3
    obb_offset = rgb_offset + rgb_len
    if len(payload) < obb_offset:
        return None

    if np is not None:
        pts = np.frombuffer(
            payload,
            dtype="<f4",
            count=n * 3,
            offset=xyz_offset,
        ).reshape(n, 3).copy()
        colors = np.frombuffer(
            payload,
            dtype=np.uint8,
            count=n * 3,
            offset=rgb_offset,
        ).reshape(n, 3).copy()
    else:
        pts = [
            struct.unpack_from("<fff", payload, xyz_offset + i * 12)
            for i in range(n)
        ]
        colors = None

    obbs, _ = parse_obb_payload(payload, obb_offset)
    pts = apply_axis_ops_array(pts)
    for obb in obbs:
        obb["corners"] = apply_axis_ops_array(obb["corners"])

    if STATE["auto_fit"] and len(pts) > 50:
        pts, fit_corners = center_and_scale_frame(pts, [obb["corners"] for obb in obbs])
        for obb, corners in zip(obbs, fit_corners):
            obb["corners"] = corners

    latency_ms = max(0.0, (time.time_ns() - timestamp_ns) / 1_000_000.0)
    return {
        "frame_id": frame_id,
        "timestamp_ns": timestamp_ns,
        "points": pts,
        "colors": colors,
        "obbs": obbs,
        "latency_ms": latency_ms,
    }


def recv_stream_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)

    buffer = bytearray()

    while STATE["running"]:
        try:
            chunk, _ = sock.recvfrom(65535)
            buffer.extend(chunk)

            if len(buffer) < 4:
                continue

            n = struct.unpack_from("<I", buffer, 0)[0]
            need = 4 + (n * 12) + (n * 3)

            if len(buffer) < need:
                continue

            if len(buffer) < need + 2:
                continue
            obb_count = struct.unpack_from("<H", buffer, need)[0]
            full_need = need + 2 + obb_count * (4 + 2 + 8 * 12)
            if len(buffer) < full_need:
                continue

            offset = 4
            xyz_bytes = buffer[offset:offset + n*12]
            offset += n*12
            # rgb_bytes = buffer[offset:offset + n*3]  # lo usaremos luego

            pts = []
            for i in range(n):
                x, y, z = struct.unpack_from("<fff", xyz_bytes, i*12)
                pts.append(apply_axis_ops((x, y, z)))

            if STATE["auto_fit"] and len(pts) > 50:
                pts = center_and_scale(pts)

            with STATE["lock"]:
                STATE["points"] = pts
                STATE["last_n"] = n

            buffer = buffer[full_need:]

        except socket.timeout:
            continue
        except Exception:
            buffer = bytearray()

    sock.close()


def recv_stream_v2_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_V2_PORT))
    sock.settimeout(1.0)

    assemblies = {}
    newest_seen = 0

    while STATE["running"]:
        try:
            packet, _ = sock.recvfrom(65535)
            if len(packet) < V2_HEADER.size:
                continue

            (
                magic,
                version,
                header_len,
                frame_id,
                timestamp_ns,
                chunk_index,
                chunk_count,
                payload_len,
                frame_len,
                crc32,
            ) = V2_HEADER.unpack_from(packet, 0)

            if magic != V2_MAGIC or version != V2_VERSION or header_len != V2_HEADER.size:
                continue
            if chunk_count == 0 or chunk_index >= chunk_count:
                continue

            chunk = packet[header_len:]
            if len(chunk) != payload_len:
                continue
            if (zlib.crc32(chunk) & 0xFFFFFFFF) != crc32:
                continue

            if frame_id > newest_seen:
                newest_seen = frame_id
                assemblies = {
                    fid: entry
                    for fid, entry in assemblies.items()
                    if fid >= frame_id
                }
            elif frame_id < newest_seen:
                continue

            entry = assemblies.setdefault(
                frame_id,
                {
                    "timestamp_ns": timestamp_ns,
                    "chunk_count": chunk_count,
                    "frame_len": frame_len,
                    "chunks": {},
                },
            )
            entry["chunks"][chunk_index] = chunk

            if len(entry["chunks"]) != entry["chunk_count"]:
                continue

            payload = b"".join(entry["chunks"][i] for i in range(entry["chunk_count"]))
            assemblies.pop(frame_id, None)
            if len(payload) != entry["frame_len"]:
                continue

            frame = decode_v2_frame(payload, frame_id, entry["timestamp_ns"])
            if frame is None:
                continue

            with STATE["lock"]:
                last_frame_id = STATE["last_frame_id"]
                if frame_id <= last_frame_id:
                    continue
                if last_frame_id and frame_id > last_frame_id + 1:
                    STATE["dropped_frames"] += int(frame_id - last_frame_id - 1)
                STATE["latest_frame"] = frame
                STATE["last_frame_id"] = frame_id
                STATE["last_n"] = len(frame["points"])
                STATE["last_latency_ms"] = frame["latency_ms"]

        except socket.timeout:
            continue
        except Exception:
            assemblies.clear()

    sock.close()


def rpc_click(point):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((TCP_IP, TCP_PORT))
    s.sendall(struct.pack("<fff", point[0], point[1], point[2]))

    ok = s.recv(1)
    if not ok:
        s.close()
        return None, None
    ok = struct.unpack("<B", ok)[0]

    if ok == 1:
        corners_b = b""
        while len(corners_b) < 96:
            corners_b += s.recv(96 - len(corners_b))

        corners = []
        for i in range(8):
            x, y, z = struct.unpack_from("<fff", corners_b, i*12)
            corners.append(apply_axis_ops((x, y, z)))

        if STATE["auto_fit"] and len(corners) == 8:
            corners = center_and_scale(corners)

        ln = struct.unpack("<H", s.recv(2))[0]
        label = s.recv(ln).decode("utf-8") if ln else "object"
        s.close()
        return corners, label
    else:
        ln = struct.unpack("<H", s.recv(2))[0]
        msg = s.recv(ln).decode("utf-8") if ln else "rpc error"
        s.close()
        return None, msg


def timer_update():
    if not STATE["running"]:
        return None

    obj = ensure_pointcloud_object()
    if STATE["stream_mode"] == "v2":
        bbox_obj = ensure_bbox_object()
        with STATE["lock"]:
            frame = STATE["latest_frame"]
            displayed_frame_id = STATE["displayed_frame_id"]

        if frame is None or frame["frame_id"] == displayed_frame_id:
            return 0.016

        pts = frame["points"]
        set_pointcloud_data(obj, pts, radius=STATE["point_radius"])

        obbs = frame["obbs"]
        if obbs:
            set_obb_from_corners(bbox_obj, obbs[0]["corners"])

        with STATE["lock"]:
            STATE["points"] = pts
            STATE["colors"] = frame["colors"]
            STATE["obbs"] = obbs
            STATE["displayed_frame_id"] = frame["frame_id"]
            STATE["last_n"] = len(pts)

        return 0.016

    with STATE["lock"]:
        pts = STATE["points"]

    if pts:
        set_pointcloud_data(obj, pts, radius=STATE["point_radius"])

    return 0.05


class BLAZE_OT_start(bpy.types.Operator):
    bl_idname = "blaze.start"
    bl_label = "Start Legacy"

    def execute(self, context):
        if STATE["running"]:
            return {'FINISHED'}
        STATE["stream_mode"] = "legacy"
        STATE["latest_frame"] = None
        STATE["displayed_frame_id"] = 0
        STATE["last_frame_id"] = 0
        STATE["last_n"] = 0
        STATE["last_latency_ms"] = 0.0
        STATE["dropped_frames"] = 0
        STATE["points"] = []
        STATE["colors"] = None
        STATE["obbs"] = []
        STATE["running"] = True
        threading.Thread(target=recv_stream_thread, daemon=True).start()
        bpy.app.timers.register(timer_update)
        self.report({'INFO'}, "Blaze legacy stream started")
        return {'FINISHED'}


class BLAZE_OT_start_v2(bpy.types.Operator):
    bl_idname = "blaze.start_v2"
    bl_label = "Start V2 Realtime"

    def execute(self, context):
        if STATE["running"]:
            return {'FINISHED'}
        STATE["stream_mode"] = "v2"
        STATE["latest_frame"] = None
        STATE["displayed_frame_id"] = 0
        STATE["last_frame_id"] = 0
        STATE["last_n"] = 0
        STATE["last_latency_ms"] = 0.0
        STATE["dropped_frames"] = 0
        STATE["points"] = []
        STATE["colors"] = None
        STATE["obbs"] = []
        STATE["running"] = True
        threading.Thread(target=recv_stream_v2_thread, daemon=True).start()
        bpy.app.timers.register(timer_update)
        self.report({'INFO'}, "Blaze v2 realtime stream started")
        return {'FINISHED'}


class BLAZE_OT_stop(bpy.types.Operator):
    bl_idname = "blaze.stop"
    bl_label = "Stop Blaze Stream"

    def execute(self, context):
        STATE["running"] = False
        self.report({'INFO'}, "Blaze stream stopped")
        return {'FINISHED'}


class BLAZE_OT_pick(bpy.types.Operator):
    bl_idname = "blaze.pick"
    bl_label = "Pick + BBox"

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Debe ser en View3D")
            return {'CANCELLED'}

        from bpy_extras import view3d_utils
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()

        with STATE["lock"]:
            pts = STATE["points"]

        if pts is None or len(pts) == 0:
            self.report({'ERROR'}, "No hay puntos aún")
            return {'CANCELLED'}

        o = origin
        d = direction
        best_p = None
        best_dist = 1e9

        # subsample para velocidad
        step = 5
        for p in pts[::step]:
            pv = Vector(p)
            v = pv - o
            t = v.dot(d)
            proj = o + d * t
            dist = (pv - proj).length
            if dist < best_dist:
                best_dist = dist
                best_p = pv

        if best_p is None:
            return {'CANCELLED'}

        corners, label = rpc_click(best_p)
        if corners is None:
            self.report({'ERROR'}, f"RPC: {label}")
            return {'CANCELLED'}

        bbox_obj = ensure_bbox_object()
        set_bbox_from_corners(bbox_obj, corners)
        self.report({'INFO'}, f"Label: {label}")
        return {'FINISHED'}


class BLAZE_PT_panel(bpy.types.Panel):
    bl_label = "Blaze"
    bl_idname = "BLAZE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Blaze'

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("blaze.start_v2")
        row.operator("blaze.start")
        row = layout.row()
        row.operator("blaze.stop")
        layout.operator("blaze.pick", text="Click → BBox + Label")

        layout.separator()
        layout.label(text=f"Mode: {STATE['stream_mode']}")
        layout.label(text=f"Last points: {STATE['last_n']}")
        layout.label(text=f"Frame: {STATE['displayed_frame_id']}")
        layout.label(text=f"Latency: {STATE['last_latency_ms']:.1f} ms")
        layout.label(text=f"Dropped: {STATE['dropped_frames']}")
        layout.label(text="Ajuste rápido:")
        layout.prop(context.scene, "blaze_point_radius")
        layout.prop(context.scene, "blaze_target_size")


def sync_scene_props(_self, _context):
    scene = getattr(_context, "scene", None)
    if scene is None:
        return
    STATE["point_radius"] = scene.blaze_point_radius
    STATE["target_size"] = scene.blaze_target_size


classes = (BLAZE_OT_start, BLAZE_OT_start_v2, BLAZE_OT_stop, BLAZE_OT_pick, BLAZE_PT_panel)

def register():
    bpy.types.Scene.blaze_point_radius = bpy.props.FloatProperty(
        name="Point Radius",
        default=0.01,
        min=0.0001,
        max=0.2,
        update=sync_scene_props
    )
    bpy.types.Scene.blaze_target_size = bpy.props.FloatProperty(
        name="Target Size",
        default=2.0,
        min=0.1,
        max=50.0,
        update=sync_scene_props
    )
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.blaze_point_radius
    del bpy.types.Scene.blaze_target_size
