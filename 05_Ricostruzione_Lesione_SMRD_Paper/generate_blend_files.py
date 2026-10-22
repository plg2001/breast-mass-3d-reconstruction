import subprocess
import os

blender_exe = r"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"
if not os.path.exists(blender_exe):
    blender_exe = r"Blender\blender.exe"

base_dir = r"c:\Users\Pierluca\Desktop\Tesi\05_Ricostruzione_Lesione_SMRD_Paper\output"

for p in ["P_00232", "P_01226", "P_01754"]:
    glb_path = os.path.join(base_dir, p, f"{p}_smrd_lesion.glb")
    blend_path = os.path.join(base_dir, p, f"{p}_smrd_lesion.blend")
    
    if not os.path.exists(glb_path):
        print(f"[SKIP] GLB not found for {p}")
        continue

    script = (
        "import bpy\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        f"bpy.ops.import_scene.gltf(filepath=r'{glb_path}')\n"
        "for obj in bpy.context.scene.objects:\n"
        "    if obj.type == 'MESH':\n"
        "        bpy.context.view_layer.objects.active = obj\n"
        "        obj.select_set(True)\n"
        "        bpy.ops.object.shade_smooth()\n"
        f"bpy.ops.wm.save_as_mainfile(filepath=r'{blend_path}')\n"
    )
    
    res = subprocess.run([blender_exe, "--background", "--python-expr", script], capture_output=True, text=True)
    if os.path.exists(blend_path):
        print(f"[OK] {p}: .blend file created ({os.path.getsize(blend_path)} bytes)")
    else:
        print(f"[ERRORE] {p}: {res.stderr}")
