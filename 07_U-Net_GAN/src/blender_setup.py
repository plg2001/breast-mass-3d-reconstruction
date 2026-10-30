"""
Script per creare una scena Blender (.blend) con materiali semitrasparenti e luci studio
a partire dal file GLB multi-strato (Cute, Areola-Capezzolo, Vene superficiali, Arterie profonde,
Grasso premammario, Parenchima, Dotti galattofori, Legamenti di Cooper, Linfonodi intramammari,
Cuscino retromammario, Muscolo pettorale).
"""

import sys
import os

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def setup_scene(glb_path: str, blend_output_path: str, render_png_path: str = None):
    if not IN_BLENDER:
        print("[ERRORE] Eseguire all'interno di Blender!")
        return

    # 1. Pulisci la scena predefinita
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Configura World Studio con illuminazione d'ambiente morbida neutra
    world = bpy.data.worlds.new("StudioWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.18, 0.19, 0.22, 1.0)
        bg.inputs["Strength"].default_value = 0.90

    # 3. Importa il modello glTF/GLB multi-layer
    print(f"[BLENDER] Importazione GLB: {glb_path}")
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # 4. Configura materiali realistici per ciascuno strato anatomico
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()

            mat = bpy.data.materials.new(name=obj.name + "_material")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")

            name_lower = obj.name.lower()
            if bsdf:
                if "skin" in name_lower:
                    # Cute / Derma: rosa carne tenue, molto trasparente
                    bsdf.inputs["Base Color"].default_value = (0.98, 0.82, 0.78, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.12
                    bsdf.inputs["Roughness"].default_value = 0.35
                    if "Subsurface Weight" in bsdf.inputs:
                        bsdf.inputs["Subsurface Weight"].default_value = 0.12
                    mat.diffuse_color = (0.98, 0.82, 0.78, 0.12)
                    
                elif "areola" in name_lower or "nipple" in name_lower or "nac" in name_lower:
                    # Complesso Areola-Capezzolo (NAC): pigmento melanico naturale bruno/rosato
                    bsdf.inputs["Base Color"].default_value = (0.72, 0.42, 0.38, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.98
                    bsdf.inputs["Roughness"].default_value = 0.55
                    mat.diffuse_color = (0.72, 0.42, 0.38, 0.98)
                    
                elif "vein" in name_lower or "venous" in name_lower:
                    # Plesso venoso superficiale (cerchio di Haller): azzurro/blu venoso
                    bsdf.inputs["Base Color"].default_value = (0.12, 0.42, 0.85, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.92
                    bsdf.inputs["Roughness"].default_value = 0.20
                    mat.diffuse_color = (0.12, 0.42, 0.85, 0.92)
                    
                elif "arter" in name_lower:
                    # Rete arteriosa ghiandolare: rosso vivo scarlatto
                    bsdf.inputs["Base Color"].default_value = (0.85, 0.12, 0.16, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.95
                    bsdf.inputs["Roughness"].default_value = 0.25
                    mat.diffuse_color = (0.85, 0.12, 0.16, 0.95)
                    
                elif "retro" in name_lower:
                    # Cuscino adiposo retromammario (spazio di Chassaignac)
                    bsdf.inputs["Base Color"].default_value = (0.96, 0.92, 0.62, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.20
                    bsdf.inputs["Roughness"].default_value = 0.60
                    mat.diffuse_color = (0.96, 0.92, 0.62, 0.20)
                    
                elif "fat" in name_lower or "adipose" in name_lower:
                    # Grasso premammario sottocutaneo: giallo avorio tenue semitrasparente
                    bsdf.inputs["Base Color"].default_value = (0.95, 0.90, 0.58, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.18
                    bsdf.inputs["Roughness"].default_value = 0.55
                    if "Subsurface Weight" in bsdf.inputs:
                        bsdf.inputs["Subsurface Weight"].default_value = 0.20
                    mat.diffuse_color = (0.95, 0.90, 0.58, 0.18)
                    
                elif "tissue" in name_lower or "parenchyma" in name_lower or "fibroglandular" in name_lower:
                    # Tessuto parenchimale fibroglandolare: arancio ambrato semitrasparente
                    bsdf.inputs["Base Color"].default_value = (0.92, 0.62, 0.38, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.45
                    bsdf.inputs["Roughness"].default_value = 0.45
                    mat.diffuse_color = (0.92, 0.62, 0.38, 0.45)
                    
                elif "duct" in name_lower or "lactiferous" in name_lower:
                    # Rete duttale galattofora (albero VICTRE): bianco latte perlaceo opaco
                    bsdf.inputs["Base Color"].default_value = (0.96, 0.97, 1.0, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.94
                    bsdf.inputs["Roughness"].default_value = 0.18
                    mat.diffuse_color = (0.96, 0.97, 1.0, 0.94)
                    
                elif "cooper" in name_lower or "ligament" in name_lower:
                    # Legamenti di Cooper (retinacula cutis): cordoni fibrosi argentei
                    bsdf.inputs["Base Color"].default_value = (0.86, 0.89, 0.95, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.85
                    bsdf.inputs["Roughness"].default_value = 0.25
                    mat.diffuse_color = (0.86, 0.89, 0.95, 0.85)
                    
                elif "lymph" in name_lower or "node" in name_lower:
                    # Linfonodi intramammari reniformi: verde oliva chiaro/buff
                    bsdf.inputs["Base Color"].default_value = (0.70, 0.76, 0.58, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.90
                    bsdf.inputs["Roughness"].default_value = 0.40
                    mat.diffuse_color = (0.70, 0.76, 0.58, 0.90)
                    
                elif "pectoral" in name_lower or "muscle" in name_lower:
                    # Muscolo grande pettorale: rosso muscolare profondo
                    bsdf.inputs["Base Color"].default_value = (0.65, 0.16, 0.16, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.85
                    bsdf.inputs["Roughness"].default_value = 0.50
                    mat.diffuse_color = (0.65, 0.16, 0.16, 0.85)

                try:
                    mat.blend_method = "BLEND"
                except Exception:
                    pass
                if hasattr(mat, "surface_render_method"):
                    try:
                        mat.surface_render_method = "BLENDED"
                    except Exception:
                        pass

            obj.data.materials.clear()
            obj.data.materials.append(mat)

    # 5. Illuminazione Studio Multi-Punto
    sun_data = bpy.data.lights.new(name="Sun_Key", type='SUN')
    sun_data.energy = 2.8
    sun_obj = bpy.data.objects.new("Sun_Key", sun_data)
    sun_obj.rotation_euler = (0.6, 0.3, 0.4)
    bpy.context.collection.objects.link(sun_obj)

    fill_data = bpy.data.lights.new(name="Light_Fill", type='POINT')
    fill_data.energy = 220.0
    fill_data.color = (0.92, 0.95, 1.0)
    fill_obj = bpy.data.objects.new("Light_Fill", fill_data)
    fill_obj.location = (-100.0, -80.0, 120.0)
    bpy.context.collection.objects.link(fill_obj)

    rim_data = bpy.data.lights.new(name="Light_Rim", type='POINT')
    rim_data.energy = 160.0
    rim_data.color = (1.0, 0.96, 0.90)
    rim_obj = bpy.data.objects.new("Light_Rim", rim_data)
    rim_obj.location = (80.0, 100.0, -30.0)
    bpy.context.collection.objects.link(rim_obj)

    # 6. Configura Telecamera
    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3
    has_mesh = False
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

        # Vista Prospettica Anterolaterale 3D (3/4 view)
        target = mathutils.Vector((center[0], center[1] + size * 0.05, center[2] + size * 0.10))
        cam_obj.location = (
            center[0] + size * 1.15,
            center[1] - size * 0.60,
            center[2] + size * 0.95
        )
        direction = target - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

    # 7. Salva la scena .blend
    os.makedirs(os.path.dirname(os.path.abspath(blend_output_path)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_output_path)
    print(f"[BLENDER] Scena salvata in: {blend_output_path}")

    # 8. Rendering Preview
    if render_png_path and has_mesh:
        os.makedirs(os.path.dirname(os.path.abspath(render_png_path)), exist_ok=True)
        bpy.context.scene.render.filepath = render_png_path
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 1024
        try:
            bpy.ops.render.render(write_still=True)
            print(f"[BLENDER] Preview 3D renderizzata in: {render_png_path}")
        except Exception as e:
            print(f"[BLENDER] Render fallito (non critico): {e}")

        # Vista 2: Profilo Anatomico Laterale MLO
        profile_png = render_png_path.replace(".png", "_profile.png")
        target_prof = mathutils.Vector((center[0], center[1], center[2]))
        cam_obj.location = (
            center[0] + size * 1.65,
            center[1],
            center[2]
        )
        direction = target_prof - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()
        bpy.context.scene.render.filepath = profile_png
        try:
            bpy.ops.render.render(write_still=True)
            print(f"[BLENDER] Profilo anatomico renderizzato in: {profile_png}")
        except Exception as e:
            pass


if __name__ == "__main__":
    args = sys.argv
    if "--" in args:
        idx = args.index("--")
        custom_args = args[idx + 1:]
        if len(custom_args) >= 2:
            r_path = custom_args[2] if len(custom_args) >= 3 else None
            setup_scene(custom_args[0], custom_args[1], r_path)
