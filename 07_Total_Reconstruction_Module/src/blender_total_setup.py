"""
Blender Studio Setup & Rendering Engine: Total Breast + SMRD Lesion
===================================================================
Configura i materiali PBR/SSS fotorealistici per tutti i 15 strati anatomici
e patologici della ricostruzione completa:
1. Cute esterna (breast_skin)
2. Complesso Areola-Capezzolo (nipple_areola)
3. Plesso venoso superficiale di Haller (breast_veins)
4. Mantello adiposo sottocutaneo volumetrico (subcutaneous_fat)
5. Nervi cutanei sensoriali (mammary_nerves)
6. Vasi linfatici di Sappey e ascellari (lymphatic_vessels)
7. Linfonodi intramammari reniformi (lymph_nodes)
8. Legamenti sospensori di Cooper (cooper_ligaments)
9. Parenchima fibroglandolare rugoso (fibroglandular_tissue)
10. Albero galattoforo e acini TDLU (lactiferous_ducts)
11. Rete arteriosa ghiandolare (breast_arteries)
12. Cuscino adiposo retromammario (retromammary_fat)
13. Fascia pettorale profonda (deep_pectoral_fascia)
14. Muscolo grande pettorale (pectoral_muscle)
15. LESIONE TUMORALE SPICOLATA SMRD (tumor_lesion) - risalto iper-vascolare opaco
"""

import sys
import os

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def setup_total_scene(glb_path: str, blend_output_path: str, render_png_path: str = None):
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
        bg.inputs["Color"].default_value = (0.16, 0.17, 0.20, 1.0)
        bg.inputs["Strength"].default_value = 0.95

    # 3. Importa il modello glTF/GLB multi-layer
    print(f"[BLENDER] Importazione GLB unificato totale: {glb_path}")
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
                # -------------------------------------------------------------
                # LESIONE TUMORALE SMRD (Evidenza patologica massima)
                # -------------------------------------------------------------
                if "tumor" in name_lower or "lesion" in name_lower or "mass" in name_lower:
                    # Maligno iper-intenso con spicole riflettenti e nucleo scuro denso
                    bsdf.inputs["Base Color"].default_value = (0.58, 0.06, 0.14, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.98
                    bsdf.inputs["Roughness"].default_value = 0.25
                    bsdf.inputs["Specular IOR Level"].default_value = 0.65 if "Specular IOR Level" in bsdf.inputs else 0.5
                    if "Subsurface Weight" in bsdf.inputs:
                        bsdf.inputs["Subsurface Weight"].default_value = 0.35
                    mat.diffuse_color = (0.58, 0.06, 0.14, 0.98)

                # -------------------------------------------------------------
                # STRATI DELLA MAMMELLA
                # -------------------------------------------------------------
                elif "skin" in name_lower:
                    # Cute / Derma: tono carne naturale semitrasparente
                    bsdf.inputs["Base Color"].default_value = (0.95, 0.82, 0.77, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.22
                    bsdf.inputs["Roughness"].default_value = 0.32
                    if "Subsurface Weight" in bsdf.inputs:
                        bsdf.inputs["Subsurface Weight"].default_value = 0.25
                    mat.diffuse_color = (0.95, 0.82, 0.77, 0.22)

                elif "areola" in name_lower or "nipple" in name_lower or "nac" in name_lower:
                    # Complesso Areola-Capezzolo (NAC)
                    bsdf.inputs["Base Color"].default_value = (0.72, 0.42, 0.38, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.98
                    bsdf.inputs["Roughness"].default_value = 0.55
                    mat.diffuse_color = (0.72, 0.42, 0.38, 0.98)

                elif "vein" in name_lower or "venous" in name_lower:
                    # Plesso venoso superficiale di Haller: azzurro/blu venoso
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

                elif "nerve" in name_lower:
                    # Nervi cutanei sensoriali (4° intercostale): giallo brillante
                    bsdf.inputs["Base Color"].default_value = (0.98, 0.90, 0.15, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.98
                    bsdf.inputs["Roughness"].default_value = 0.20
                    mat.diffuse_color = (0.98, 0.90, 0.15, 0.98)

                elif "lymphatic" in name_lower or "sappey" in name_lower:
                    # Vasi linfatici di Sappey e ascellari: verde-oro luminescente
                    bsdf.inputs["Base Color"].default_value = (0.82, 0.92, 0.45, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.92
                    bsdf.inputs["Roughness"].default_value = 0.25
                    mat.diffuse_color = (0.82, 0.92, 0.45, 0.92)

                elif "fascia" in name_lower:
                    # Fascia pettorale profonda: velo fibroso perlaceo
                    bsdf.inputs["Base Color"].default_value = (0.90, 0.93, 0.98, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.38
                    bsdf.inputs["Roughness"].default_value = 0.45
                    mat.diffuse_color = (0.90, 0.93, 0.98, 0.38)

                elif "retro" in name_lower:
                    # Cuscino adiposo retromammario (spazio di Chassaignac)
                    bsdf.inputs["Base Color"].default_value = (0.95, 0.89, 0.58, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.48
                    bsdf.inputs["Roughness"].default_value = 0.50
                    mat.diffuse_color = (0.95, 0.89, 0.58, 0.48)

                elif "fat" in name_lower or "adipose" in name_lower:
                    # Grasso premammario sottocutaneo volumetrico compatto (elimina vuoti)
                    bsdf.inputs["Base Color"].default_value = (0.96, 0.90, 0.60, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.52
                    bsdf.inputs["Roughness"].default_value = 0.45
                    if "Subsurface Weight" in bsdf.inputs:
                        bsdf.inputs["Subsurface Weight"].default_value = 0.30
                    mat.diffuse_color = (0.96, 0.90, 0.60, 0.52)

                elif "tissue" in name_lower or "parenchyma" in name_lower or "fibroglandular" in name_lower:
                    # Tessuto parenchimale fibroglandolare rugoso U-Net-GAN
                    bsdf.inputs["Base Color"].default_value = (0.90, 0.60, 0.40, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.42
                    bsdf.inputs["Roughness"].default_value = 0.45
                    mat.diffuse_color = (0.90, 0.60, 0.40, 0.42)

                elif "duct" in name_lower or "lactiferous" in name_lower:
                    # Rete duttale galattofora e acini TDLU: bianco latte perlaceo
                    bsdf.inputs["Base Color"].default_value = (0.96, 0.97, 1.0, 1.0)
                    bsdf.inputs["Alpha"].default_value = 0.94
                    bsdf.inputs["Roughness"].default_value = 0.18
                    mat.diffuse_color = (0.96, 0.97, 1.0, 0.94)

                elif "cooper" in name_lower or "ligament" in name_lower:
                    # Legamenti di Cooper: cordoni fibrosi argentei
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

    # 5. Configura Luci Studio a 3 Punti
    light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    light_data.energy = 450.0
    light_data.size = 200.0
    light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (150.0, -180.0, 220.0)

    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 220.0
    fill_data.size = 250.0
    fill_obj = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (-180.0, -140.0, 160.0)

    rim_data = bpy.data.lights.new(name="RimLight", type='POINT')
    rim_data.energy = 380.0
    rim_obj = bpy.data.objects.new(name="RimLight", object_data=rim_data)
    bpy.context.scene.collection.objects.link(rim_obj)
    rim_obj.location = (0.0, 160.0, 220.0)

    # 6. Calcola Bounding Box e posiziona Telecamera
    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3
    has_mesh = False

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            has_mesh = True
            for corner in obj.bound_box:
                world_co = obj.matrix_world @ mathutils.Vector(corner)
                for i in range(3):
                    min_co[i] = min(min_co[i], world_co[i])
                    max_co[i] = max(max_co[i], world_co[i])

    cam_data = bpy.data.cameras.new("MainCamera")
    cam_data.lens = 65.0
    cam_data.clip_start = 0.5
    cam_data.clip_end = 3000.0
    cam_obj = bpy.data.objects.new("MainCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    if has_mesh:
        center = [(min_co[i] + max_co[i]) / 2.0 for i in range(3)]
        size = max(max_co[i] - min_co[i] for i in range(3))

        target = mathutils.Vector((center[0], center[1], center[2]))
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
    print(f"[BLENDER] Scena totale salvata in: {blend_output_path}")

    # 8. Rendering Preview
    if render_png_path and has_mesh:
        os.makedirs(os.path.dirname(os.path.abspath(render_png_path)), exist_ok=True)
        bpy.context.scene.render.filepath = render_png_path
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 1024
        try:
            bpy.ops.render.render(write_still=True)
            print(f"[BLENDER] Preview 3D totale renderizzata in: {render_png_path}")
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
            glb = custom_args[0]
            blend = custom_args[1]
            render = custom_args[2] if len(custom_args) > 2 else None
            setup_total_scene(glb, blend, render)
