"""
Script di Automazione Blender per Visualizzazione e Rendering 3D
Eseguito all'interno dell'ambiente Python di Blender.
Orientamento anatomico verticale: Seno eretto con proiezione anteriore.
"""

import sys
import os

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def setup_blender_scene(
    glb_path: str,
    blend_output_path: str,
    preview_png_path: str
):
    if not IN_BLENDER:
        print("[ERRORE] Questo script deve essere eseguito all'interno di Blender!")
        return

    # 1. Pulisci la scena predefinita
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Importa il modello glTF/GLB
    print(f"[BLENDER] Importazione file: {glb_path}")
    bpy.ops.import_scene.gltf(filepath=glb_path)

    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objs:
        print("[BLENDER] Nessuna mesh trovata nel file GLB!")
        return

    for obj in mesh_objs:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

    # 3. Calcola il bounding box globale della scena per inquadrare la camera
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
    max_dim = max(dimensions.x, dimensions.y, dimensions.z, 50.0)

    # 4. Configura illuminazione studio a 3 punti per seno eretto verticale
    # Key Light (Luce principale calda in alto a destra)
    key_light_data = bpy.data.lights.new(name="Key_Light", type='AREA')
    key_light_data.energy = 600.0
    key_light_data.size = max_dim * 0.8
    key_light_data.color = (1.0, 0.96, 0.92)
    key_light = bpy.data.objects.new("Key_Light", key_light_data)
    key_light.location = center + mathutils.Vector((max_dim * 1.4, -max_dim * 1.8, max_dim * 1.2))
    bpy.context.collection.objects.link(key_light)

    # Fill Light (Luce di riempimento soffusa a sinistra)
    fill_light_data = bpy.data.lights.new(name="Fill_Light", type='AREA')
    fill_light_data.energy = 300.0
    fill_light_data.size = max_dim * 1.2
    fill_light_data.color = (0.92, 0.95, 1.0)
    fill_light = bpy.data.objects.new("Fill_Light", fill_light_data)
    fill_light.location = center + mathutils.Vector((-max_dim * 1.6, -max_dim * 1.5, max_dim * 0.4))
    bpy.context.collection.objects.link(fill_light)

    # Rim Light (Controluce posteriore per evidenziare la sagoma)
    rim_light_data = bpy.data.lights.new(name="Rim_Light", type='POINT')
    rim_light_data.energy = 450.0
    rim_light_data.color = (1.0, 1.0, 1.0)
    rim_light = bpy.data.objects.new("Rim_Light", rim_light_data)
    rim_light.location = center + mathutils.Vector((0.0, max_dim * 1.8, max_dim * 1.5))
    bpy.context.collection.objects.link(rim_light)

    # 5. Fotocamera frontale-obliqua (prospettiva 3/4 naturale)
    cam_data = bpy.data.cameras.new("Main_Camera")
    cam_data.lens = 75.0
    cam = bpy.data.objects.new("Main_Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Inquadratura frontale-obliqua
    cam_pos = center + mathutils.Vector((max_dim * 1.3, -max_dim * 2.3, max_dim * 0.5))
    cam.location = cam_pos

    # Orienta la camera verso il centro del seno
    direction = center - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    # 6. Configura impostazioni di rendering
    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    # Sfondo grigio scuro neutro elegante
    world = bpy.data.worlds.new("Studio_World")
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.08, 0.09, 0.11, 1.0)
        bg_node.inputs['Strength'].default_value = 0.8
    scene.world = world

    # Salva il file .blend pronto per l'utente
    os.makedirs(os.path.dirname(blend_output_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_output_path)
    print(f"[BLENDER] File di scena salvato: {blend_output_path}")

    # Render anteprima fotografica
    scene.render.filepath = preview_png_path
    bpy.ops.render.render(write_still=True)
    print(f"[BLENDER] Render anteprima completato: {preview_png_path}")


if __name__ == "__main__":
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) >= 3:
            glb = args[0]
            blend = args[1]
            png = args[2]
            setup_blender_scene(glb, blend, png)
        else:
            print("[BLENDER] Argomenti insufficienti per lo script!")
