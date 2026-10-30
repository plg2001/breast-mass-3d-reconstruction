"""
Script per creare una scena Blender (.blend) con materiali semitrasparenti e luci
a partire dal file GLB generato da 3D U-Net-GAN.
"""

import sys
import os

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def setup_scene(glb_path: str, blend_output_path: str):
    if not IN_BLENDER:
        print("[ERRORE] Eseguire all'interno di Blender!")
        return

    # 1. Pulisci la scena predefinita
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Importa il modello glTF/GLB
    print(f"[BLENDER] Importazione GLB: {glb_path}")
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # 3. Configura materiali e trasparenze
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # Shade smooth
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()

            mat = bpy.data.materials.new(name=obj.name + "_material")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")

            name_lower = obj.name.lower()
            if bsdf:
                if "skin" in name_lower:
                    # Cute / Pelle: rosa tenue, molto trasparente
                    bsdf.inputs["Base Color"].default_value = (1.0, 0.80, 0.76, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.22
                    bsdf.inputs["Roughness"].default_value = 0.4
                    try:
                        mat.blend_method = "BLEND"
                    except Exception:
                        pass
                    if hasattr(mat, "surface_render_method"):
                        try:
                            mat.surface_render_method = "BLENDED"
                        except Exception:
                            pass
                    mat.diffuse_color = (1.0, 0.80, 0.76, 0.22)
                elif "tissue" in name_lower:
                    # Tessuto fibroglandolare interno: arancio/rosa, semitrasparente
                    bsdf.inputs["Base Color"].default_value = (0.95, 0.65, 0.40, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.45
                    bsdf.inputs["Roughness"].default_value = 0.5
                    try:
                        mat.blend_method = "BLEND"
                    except Exception:
                        pass
                    if hasattr(mat, "surface_render_method"):
                        try:
                            mat.surface_render_method = "BLENDED"
                        except Exception:
                            pass
                    mat.diffuse_color = (0.95, 0.65, 0.40, 0.45)
                elif "lesion" in name_lower:
                    # Lesione tumorale: rosso brillante solido
                    bsdf.inputs["Base Color"].default_value = (0.95, 0.05, 0.05, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.2
                    mat.diffuse_color = (0.95, 0.05, 0.05, 1.0)
                    if "Emission Color" in bsdf.inputs:
                        bsdf.inputs["Emission Color"].default_value = (0.5, 0.02, 0.02, 1.0)
                    elif "Emission" in bsdf.inputs:
                        bsdf.inputs["Emission"].default_value = (0.5, 0.02, 0.02, 1.0)
                    if "Emission Strength" in bsdf.inputs:
                        bsdf.inputs["Emission Strength"].default_value = 1.0

            obj.data.materials.clear()
            obj.data.materials.append(mat)

    # 4. Configura illuminazione a 3 punti
    # Luce Sole / Chiave
    sun_data = bpy.data.lights.new(name="Sun_Key", type='SUN')
    sun_data.energy = 2.5
    sun_obj = bpy.data.objects.new("Sun_Key", sun_data)
    sun_obj.rotation_euler = (0.7, 0.2, 0.5)
    bpy.context.collection.objects.link(sun_obj)

    # Luce Fill soffusa
    fill_data = bpy.data.lights.new(name="Light_Fill", type='POINT')
    fill_data.energy = 200.0
    fill_data.color = (0.9, 0.95, 1.0)
    fill_obj = bpy.data.objects.new("Light_Fill", fill_data)
    fill_obj.location = (-50.0, -50.0, 50.0)
    bpy.context.collection.objects.link(fill_obj)

    # 5. Configura Telecamera puntata sul modello
    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3
    has_mesh = False
    import mathutils
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            has_mesh = True
            for v in obj.bound_box:
                world_v = obj.matrix_world @ mathutils.Vector(v)
                for i in range(3):
                    min_co[i] = min(min_co[i], world_v[i])
                    max_co[i] = max(max_co[i], world_v[i])

    if has_mesh:
        center = [(min_co[i] + max_co[i]) / 2.0 for i in range(3)]
        size = max(max_co[i] - min_co[i] for i in range(3))
        cam_data = bpy.data.cameras.new(name="MainCamera")
        cam_obj = bpy.data.objects.new("MainCamera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj

        # Posiziona la camera di lato/frontale a distanza proporzionata
        cam_obj.location = (center[0] + size * 1.6, center[1] - size * 1.8, center[2] + size * 0.7)
        direction = mathutils.Vector(center) - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

    # 6. Imposta la vista 3D in modalita Material Preview
    try:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.type = 'MATERIAL'
    except Exception:
        pass

    # 7. Salva il file .blend
    os.makedirs(os.path.dirname(os.path.abspath(blend_output_path)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_output_path)
    print(f"[BLENDER] Scena salvata in: {blend_output_path}")


if __name__ == "__main__":
    args = sys.argv
    if "--" in args:
        idx = args.index("--")
        custom_args = args[idx + 1:]
        if len(custom_args) >= 2:
            setup_scene(custom_args[0], custom_args[1])
