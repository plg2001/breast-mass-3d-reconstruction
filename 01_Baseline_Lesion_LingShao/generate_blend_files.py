"""
Utility per generare file .blend (scene native Blender complete di luci, materiali e camera)
a partire dai file .glb esportati nella cartella output/ di Ling Shao.
"""

import subprocess
import os
from pathlib import Path

blender_candidates = [
    Path(r"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"),
    Path.home() / "OneDrive" / "Documenti" / "Blender" / "blender.exe",
    Path("Blender/blender.exe"),
    Path("../Blender/blender.exe"),
    Path(__file__).resolve().parent / "Blender" / "blender.exe",
    Path(__file__).resolve().parent.parent / "Blender" / "blender.exe",
]
blender_exe = next((b for b in blender_candidates if b.exists()), None)

if not blender_exe:
    print("[ERRORE] Eseguibile Blender non trovato!")
    exit(1)

output_dir = Path(__file__).resolve().parent / "output"
patient_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("P_")]

if not patient_dirs:
    print(f"[INFO] Nessuna cartella paziente trovata in {output_dir}")
    exit(0)

print(f"Generazione file .blend con Blender: {blender_exe}")
print("=" * 60)

for p_dir in patient_dirs:
    glb_file = p_dir / "meshes" / "lesion_reconstructed.glb"
    blend_file = p_dir / "meshes" / "lesion_reconstructed.blend"
    
    if not glb_file.exists():
        print(f"[SKIP] Nessun .glb in {glb_file}")
        continue

    script = f'''import bpy
import mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"{glb_file.resolve()}")

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
for obj in mesh_objs:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    
    mat = bpy.data.materials.new(name="Lesion_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.85, 0.42, 0.42, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.35
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

if mesh_objs:
    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3
    for obj in mesh_objs:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                min_co[i] = min(min_co[i], world_co[i])
                max_co[i] = max(max_co[i], world_co[i])
    center = mathutils.Vector([(min_co[i] + max_co[i]) * 0.5 for i in range(3)])
    dimensions = mathutils.Vector([(max_co[i] - min_co[i]) for i in range(3)])
    max_dim = max(dimensions.x, dimensions.y, dimensions.z, 10.0)

    key_light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_light_data.energy = 500.0
    key_light_data.size = max_dim * 1.5
    key_light = bpy.data.objects.new("KeyLight", key_light_data)
    key_light.location = center + mathutils.Vector((max_dim * 1.5, -max_dim * 2.0, max_dim * 1.5))
    bpy.context.collection.objects.link(key_light)

    fill_light_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_light_data.energy = 250.0
    fill_light_data.size = max_dim * 2.0
    fill_light = bpy.data.objects.new("FillLight", fill_light_data)
    fill_light.location = center + mathutils.Vector((-max_dim * 1.8, -max_dim * 1.5, max_dim * 0.5))
    bpy.context.collection.objects.link(fill_light)

    rim_light_data = bpy.data.lights.new(name="RimLight", type='POINT')
    rim_light_data.energy = 350.0
    rim_light = bpy.data.objects.new("RimLight", rim_light_data)
    rim_light.location = center + mathutils.Vector((0.0, max_dim * 2.0, max_dim * 1.2))
    bpy.context.collection.objects.link(rim_light)

    cam_data = bpy.data.cameras.new("MainCamera")
    cam_data.lens = 65.0
    cam = bpy.data.objects.new("MainCamera", cam_data)
    cam.location = center + mathutils.Vector((max_dim * 1.2, -max_dim * 2.2, max_dim * 0.8))
    direction = center - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

bpy.ops.wm.save_as_mainfile(filepath=r"{blend_file.resolve()}")
'''
    res = subprocess.run([str(blender_exe), "--background", "--python-expr", script], capture_output=True, text=True)
    if blend_file.exists():
        print(f"[OK] {p_dir.name}: generato {blend_file.name} ({blend_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[ERRORE] {p_dir.name}: {res.stderr}")

print("=" * 60)
print("Completato!")
