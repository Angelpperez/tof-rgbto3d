bl_info = {
    "name": "Blaze Realtime PointCloud Viewer",
    "author": "Angel + ChatGPT",
    "version": (0, 3, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Blaze",
    "category": "3D View",
}

import bpy
import threading
import socket
import struct
from mathutils import Vector

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

TCP_IP = "127.0.0.1"
TCP_PORT = 6006

STATE = {
    "running": False,
    "points": [],
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


def set_pointcloud_data(obj, pts, radius=0.01):
    pc = obj.data
    n = len(pts)
    pc.points.clear()
    pc.points.add(n)

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

            buffer = buffer[need:]

        except socket.timeout:
            continue
        except Exception:
            buffer = bytearray()

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
    with STATE["lock"]:
        pts = STATE["points"]

    if pts:
        set_pointcloud_data(obj, pts, radius=STATE["point_radius"])

    return 0.05


class BLAZE_OT_start(bpy.types.Operator):
    bl_idname = "blaze.start"
    bl_label = "Start Blaze Stream"

    def execute(self, context):
        if STATE["running"]:
            return {'FINISHED'}
        STATE["running"] = True
        threading.Thread(target=recv_stream_thread, daemon=True).start()
        bpy.app.timers.register(timer_update)
        self.report({'INFO'}, "Blaze stream started")
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

        if not pts:
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
        row.operator("blaze.start")
        row.operator("blaze.stop")
        layout.operator("blaze.pick", text="Click → BBox + Label")

        layout.separator()
        layout.label(text=f"Last points: {STATE['last_n']}")
        layout.label(text="Ajuste rápido:")
        layout.prop(context.scene, "blaze_point_radius")
        layout.prop(context.scene, "blaze_target_size")


def sync_scene_props(_self, _context):
    STATE["point_radius"] = bpy.context.scene.blaze_point_radius
    STATE["target_size"] = bpy.context.scene.blaze_target_size


classes = (BLAZE_OT_start, BLAZE_OT_stop, BLAZE_OT_pick, BLAZE_PT_panel)

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
    sync_scene_props(None, None)

    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.blaze_point_radius
    del bpy.types.Scene.blaze_target_size
