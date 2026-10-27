#!/usr/bin/env python3
"""
========================================================================================
Ricostruzione 3D Anatomica della Mammella (Skin Envelope) da Mammografie CC ed MLO
========================================================================================
Genera una superficie 3D esterna fotorealistica, anatomica e priva di artefatti:
- Supporto per qualunque paziente (ID selezionabile da riga di comando o interattivo)
- Rilevamento automatico di vista Destra (R-CC/R-MLO) o Sinistra (L-CC/L-MLO)
- Profilo a goccia organico estratto via ray-marching dal capezzolo (zero vuoto interno)
- Proporzioni anatomiche realistiche (Larghezza ~15-19 cm, Altezza ~16-20 cm, Profondità ~10 cm)
- Base retromammaria chiusa a cupola convessa morbida (come nel Riquadro 6 del paper)
- Piani radiografici ortogonali 2D (CC ed MLO) perfettamente aderenti all'involucro cutaneo
- Shading Alabaster Ghost semitrasparente con reticolo Wireframe sottile
- 5 Telecamere cliniche pre-configurate conformi alla figura di riferimento del paper
"""

import os
import sys
import argparse
import json
import time
import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

BLENDER_EXE = r"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"

def find_mammogram_pair(data_dir, patient_id):
    """
    Individua automaticamente la coppia di immagini CC ed MLO per il paziente specificato,
    supportando sia seno destro (R) che sinistro (L) e vari formati (.jpg, .png, .dcm).
    """
    candidate_dirs = [
        os.path.join(data_dir, patient_id),
        os.path.join("..", data_dir, patient_id),
        os.path.join(os.path.dirname(__file__), "..", data_dir, patient_id)
    ]
    patient_dir = None
    for cd in candidate_dirs:
        if os.path.isdir(cd):
            patient_dir = cd
            break

    if patient_dir is None:
        raise FileNotFoundError(f"Cartella del paziente '{patient_id}' non trovata in {data_dir}")

    # Priorità 1: file standard R o L (.jpg, .png, .dcm)
    for ext in [".jpg", ".png", ".dcm", ".jpeg"]:
        for lat in ["R", "L"]:
            cc = os.path.join(patient_dir, f"{lat}-CC{ext}")
            mlo = os.path.join(patient_dir, f"{lat}-MLO{ext}")
            if os.path.exists(cc) and os.path.exists(mlo):
                return lat, os.path.abspath(cc), os.path.abspath(mlo)

    # Priorità 2: ricerca flessibile per nome file (evitando le ROI)
    files = os.listdir(patient_dir)
    cc_files = [f for f in files if "cc" in f.lower() and not f.lower().startswith("roi")]
    mlo_files = [f for f in files if "mlo" in f.lower() and not f.lower().startswith("roi")]

    if cc_files and mlo_files:
        lat = "R" if "r-" in cc_files[0].lower() else ("L" if "l-" in cc_files[0].lower() else "R")
        return lat, os.path.abspath(os.path.join(patient_dir, cc_files[0])), os.path.abspath(os.path.join(patient_dir, mlo_files[0]))

    raise FileNotFoundError(f"Nessuna coppia di mammografie CC + MLO trovata per il paziente '{patient_id}' in {patient_dir}")

def extract_single_view(image_path):
    """
    Estrae l'immagine ridimensionata, la maschera del tessuto, l'orientamento della parete
    toracica e la posizione automatica del capezzolo.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossibile aprire l'immagine: {image_path}")

    h_orig, w_orig = img.shape
    scale = 600.0 / max(h_orig, w_orig)
    small = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
    h_s, w_s = small.shape

    # Pulizia bordi
    clean = small.copy()
    clean[:15, :] = 0; clean[-15:, :] = 0; clean[:, :15] = 0; clean[:, -15:] = 0

    _, bin_img = cv2.threshold(clean, 20, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8)

    is_chest_right = mask[:, 3 * w_s // 4:].mean() > mask[:, :w_s // 4].mean()

    # Identificazione del capezzolo (colonna più lontana dalla parete toracica)
    cols_tissue = []
    for r in range(h_s):
        c = np.where(mask[r, :] > 0)[0]
        if len(c) > 10:
            cols_tissue.append((r, c.min() if is_chest_right else c.max()))
    cols_tissue = np.array(cols_tissue)

    if is_chest_right:
        nip_idx = np.argmin(cols_tissue[:, 1])
    else:
        nip_idx = np.argmax(cols_tissue[:, 1])

    nip_r, nip_c = int(cols_tissue[nip_idx, 0]), int(cols_tissue[nip_idx, 1])

    return small, mask, is_chest_right, nip_r, nip_c

def extract_organic_profiles(cc_path, mlo_path, target_depth_mm=105.0, laterality="R"):
    """
    Estrae i profili cutanei anatomici tramite ray-marching radiale a partire dal capezzolo,
    funzionando correttamente sia per seni destri (R) che sinistri (L).
    """
    s_cc, m_cc, chest_right_cc, nip_r_cc, nip_c_cc = extract_single_view(cc_path)
    s_mlo, m_mlo, chest_right_mlo, nip_r_mlo, nip_c_mlo = extract_single_view(mlo_path)

    depth_px = 160
    mm_per_px = target_depth_mm / float(depth_px)

    # Ray-marching MLO (d = 0 base retromammaria, d = depth_px capezzolo)
    top_mlo, bot_mlo = [], []
    for d in range(depth_px + 1):
        step_from_nip = depth_px - d
        col = (nip_c_mlo + step_from_nip) if chest_right_mlo else (nip_c_mlo - step_from_nip)
        col = int(np.clip(col, 0, s_mlo.shape[1] - 1))
        col_vals = s_mlo[:, col]

        b = nip_r_mlo
        while b < s_mlo.shape[0] - 15 and col_vals[b] > 20:
            b += 1

        t = nip_r_mlo
        limit_t = max(25, nip_r_mlo - int(1.15 * (b - nip_r_mlo)))
        while t > limit_t and col_vals[t] > 20:
            t -= 1

        top_mlo.append(t)
        bot_mlo.append(b)

    # Ray-marching CC
    lat_cc, med_cc = [], []
    for d in range(depth_px + 1):
        step_from_nip = depth_px - d
        col = (nip_c_cc + step_from_nip) if chest_right_cc else (nip_c_cc - step_from_nip)
        col = int(np.clip(col, 0, s_cc.shape[1] - 1))
        col_vals = s_cc[:, col]

        # In CC: righe inferiori = mediale, righe superiori = laterale
        m = nip_r_cc
        while m < s_cc.shape[0] - 15 and col_vals[m] > 20:
            m += 1

        l = nip_r_cc
        limit_l = max(25, nip_r_cc - int(1.10 * (m - nip_r_cc)))
        while l > limit_l and col_vals[l] > 20:
            l -= 1

        lat_cc.append(l)
        med_cc.append(m)

    top_mlo = gaussian_filter1d(np.array(top_mlo, dtype=float), 3.5)
    bot_mlo = gaussian_filter1d(np.array(bot_mlo, dtype=float), 3.5)
    lat_cc = gaussian_filter1d(np.array(lat_cc, dtype=float), 3.5)
    med_cc = gaussian_filter1d(np.array(med_cc, dtype=float), 3.5)

    return {
        'orig_cc_path': os.path.abspath(cc_path),
        'orig_mlo_path': os.path.abspath(mlo_path),
        'laterality': laterality,
        's_cc_shape': s_cc.shape,
        's_mlo_shape': s_mlo.shape,
        'chest_right_cc': chest_right_cc,
        'chest_right_mlo': chest_right_mlo,
        'nip_r_mlo': nip_r_mlo, 'nip_c_mlo': nip_c_mlo,
        'nip_r_cc': nip_r_cc, 'nip_c_cc': nip_c_cc,
        'depth_px': depth_px,
        'mm_per_px': mm_per_px,
        'target_depth_mm': target_depth_mm,
        'top_mlo': top_mlo, 'bot_mlo': bot_mlo,
        'lat_cc': lat_cc, 'med_cc': med_cc
    }

def build_teardrop_anatomical_mesh(data, num_slices=90, num_radial=64):
    """
    Costruisce la mesh 3D anatomica a goccia, con chiusura morbida della base retromammaria
    e proporzioni antropometriche realistiche di un seno naturale.
    """
    depth_mm = data['target_depth_mm']
    mm_per_px = data['mm_per_px']
    depth_px = data['depth_px']
    is_left = (data['laterality'] == "L")

    t_samples = np.linspace(0.0, 1.0, num_slices)

    # Chiusura a cupola convessa posteriore per t in [0.0, 0.22] (spazio retromammario naturale)
    post_cap = np.where(t_samples < 0.22, np.sin((t_samples / 0.22) * (np.pi / 2.0)), 1.0)

    # Fattore di decompressione fisiologica
    decomp_scale = 0.76

    idx = np.clip((t_samples * depth_px).astype(int), 0, depth_px)

    # Segno per orientamento asse X laterale/mediale
    x_sign = -1.0 if is_left else 1.0

    # CC: asse X (Mediale / Laterale)
    X_lat = x_sign * (data['nip_r_cc'] - data['lat_cc'][idx]) * mm_per_px * post_cap * decomp_scale
    X_med = x_sign * (data['nip_r_cc'] - data['med_cc'][idx]) * mm_per_px * post_cap * decomp_scale
    if is_left:
        X_lat, X_med = X_med, X_lat

    Xc_vals = 0.5 * (X_med + X_lat)
    W_vals = np.maximum(2.0, np.abs(X_lat - X_med))

    # MLO: asse Z (Superiore / Inferiore)
    Z_sup = (data['nip_r_mlo'] - data['top_mlo'][idx]) * mm_per_px * post_cap * decomp_scale
    Z_inf = (data['nip_r_mlo'] - data['bot_mlo'][idx]) * mm_per_px * post_cap * decomp_scale
    Zc_vals = 0.5 * (Z_sup + Z_inf)
    H_vals = np.maximum(2.0, Z_sup - Z_inf)

    # Asse Y: profondità dalla base (Y=0) al capezzolo (Y=depth_mm)
    Y_vals = t_samples * depth_mm

    thetas = np.linspace(0, 2 * np.pi, num_radial, endpoint=False)
    cos_th = np.cos(thetas)
    sin_th = np.sin(thetas)
    p_exp = 2.05
    cos_sign = np.sign(cos_th) * (np.abs(cos_th) ** (2.0 / p_exp))
    sin_sign = np.sign(sin_th) * (np.abs(sin_th) ** (2.0 / p_exp))

    vertices = []
    for i in range(num_slices):
        t = t_samples[i]
        y = Y_vals[i]
        xc = Xc_vals[i]
        zc = Zc_vals[i]
        w = W_vals[i]
        h = H_vals[i]

        # Cupola convessa posteriore dolce
        dome_offset = -7.0 * (1.0 - post_cap[i])

        for j in range(num_radial):
            x = xc + 0.5 * w * cos_sign[j]
            z = zc + 0.5 * h * sin_sign[j]
            vertices.append([x, y + dome_offset, z])

    chest_center_idx = len(vertices)
    vertices.append([Xc_vals[0], -7.0, Zc_vals[0]])

    nipple_tip_idx = len(vertices)
    vertices.append([Xc_vals[-1], depth_mm, Zc_vals[-1]])

    vertices = np.array(vertices, dtype=float)
    faces = []

    for i in range(num_slices - 1):
        r_curr = i * num_radial
        r_next = (i + 1) * num_radial
        for j in range(num_radial):
            j_next = (j + 1) % num_radial
            faces.append([r_curr + j, r_curr + j_next, r_next + j_next])
            faces.append([r_curr + j, r_next + j_next, r_next + j])

    for j in range(num_radial):
        j_next = (j + 1) % num_radial
        faces.append([chest_center_idx, j_next, j])

    last_ring = (num_slices - 1) * num_radial
    for j in range(num_radial):
        j_next = (j + 1) % num_radial
        faces.append([last_ring + j, last_ring + j_next, nipple_tip_idx])

    faces = np.array(faces, dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()

    bounds_cc = (float(np.min(X_med)), float(np.max(X_lat)), float(depth_mm))
    bounds_mlo = (float(np.min(Z_inf)), float(np.max(Z_sup)), float(depth_mm))

    return mesh, bounds_cc, bounds_mlo

def generate_blender_scene(blend_path, mesh, data, bounds_cc, bounds_mlo):
    """
    Costruisce la scena Blender completa e aderente al tessuto radiografico.
    """
    verts_list = mesh.vertices.tolist()
    faces_list = mesh.faces.tolist()

    x_med_chest, x_lat_chest, depth_mm = bounds_cc
    z_inf_chest, z_sup_chest, _ = bounds_mlo

    h_c, w_c = data['s_cc_shape']
    step_c = data['depth_px']
    base_c_cc = (data['nip_c_cc'] + step_c) if data['chest_right_cc'] else (data['nip_c_cc'] - step_c)
    u_chest_cc = float(base_c_cc) / float(w_c)
    u_nip_cc = float(data['nip_c_cc']) / float(w_c)
    v_lat_cc = 1.0 - (float(data['lat_cc'][0]) / float(h_c))
    v_med_cc = 1.0 - (float(data['med_cc'][0]) / float(h_c))

    h_m, w_m = data['s_mlo_shape']
    base_c_mlo = (data['nip_c_mlo'] + step_c) if data['chest_right_mlo'] else (data['nip_c_mlo'] - step_c)
    u_chest_mlo = float(base_c_mlo) / float(w_m)
    u_nip_mlo = float(data['nip_c_mlo']) / float(w_m)
    v_sup_mlo = 1.0 - (float(data['top_mlo'][0]) / float(h_m))
    v_inf_mlo = 1.0 - (float(data['bot_mlo'][0]) / float(h_m))

    preview_img_path = os.path.abspath(blend_path.replace(".blend", "_preview.png")).replace("\\", "/")

    script = f"""
import bpy
import mathutils
import os

bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Mesh Cutanea 3D a Goccia Naturale (Alabaster Ghost)
mesh_data = bpy.data.meshes.new("mesh_breast_skin")
mesh_data.from_pydata({verts_list}, [], {faces_list})
mesh_data.polygons.foreach_set('use_smooth', [True] * len(mesh_data.polygons))
mesh_data.update()

skin_obj = bpy.data.objects.new("breast_skin_surface", mesh_data)
bpy.context.collection.objects.link(skin_obj)

mat_skin = bpy.data.materials.new(name="Mat_Breast_Skin")
mat_skin.blend_method = 'BLEND'
mat_skin.diffuse_color = (0.91, 0.89, 0.87, 0.28)
bsdf = mat_skin.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.93, 0.91, 0.89, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.20
    bsdf.inputs['IOR'].default_value = 1.33
    if 'Alpha' in bsdf.inputs:
        bsdf.inputs['Alpha'].default_value = 0.28
    if 'Transmission Weight' in bsdf.inputs:
        bsdf.inputs['Transmission Weight'].default_value = 0.40
skin_obj.data.materials.append(mat_skin)

# 2. Reticolo Wireframe Anatomico Sottile
wire_mesh = mesh_data.copy()
wire_obj = bpy.data.objects.new("breast_skin_wireframe", wire_mesh)
mod_wire = wire_obj.modifiers.new(name="Wireframe", type='WIREFRAME')
mod_wire.thickness = 0.30
mod_wire.use_replace = True

mat_wire = bpy.data.materials.new(name="Mat_Wireframe")
mat_wire.blend_method = 'BLEND'
bsdf_wire = mat_wire.node_tree.nodes.get("Principled BSDF")
bsdf_wire.inputs['Base Color'].default_value = (0.82, 0.90, 1.0, 1.0)
bsdf_wire.inputs['Alpha'].default_value = 0.38
if 'Emission Color' in bsdf_wire.inputs:
    bsdf_wire.inputs['Emission Color'].default_value = (0.82, 0.90, 1.0, 1.0)
    bsdf_wire.inputs['Emission Strength'].default_value = 0.25
wire_obj.data.materials.append(mat_wire)
bpy.context.collection.objects.link(wire_obj)

# 3. Piano Orizzontale CC (Z = 0)
verts_cc = [
    mathutils.Vector(({x_med_chest}, -5.0, 0.0)),
    mathutils.Vector(({x_lat_chest}, -5.0, 0.0)),
    mathutils.Vector(({x_lat_chest}, {depth_mm + 4.0}, 0.0)),
    mathutils.Vector(({x_med_chest}, {depth_mm + 4.0}, 0.0))
]
mesh_cc = bpy.data.meshes.new("mesh_slice_cc")
mesh_cc.from_pydata(verts_cc, [], [(0, 1, 2, 3)])
uv_cc = mesh_cc.uv_layers.new(name="UVMap")

obj_cc = bpy.data.objects.new("slice_cc_plane", mesh_cc)
bpy.context.collection.objects.link(obj_cc)

uv_data_cc = [
    ({u_chest_cc}, {v_med_cc}),
    ({u_chest_cc}, {v_lat_cc}),
    ({u_nip_cc}, {v_lat_cc}),
    ({u_nip_cc}, {v_med_cc})
]
for loop_idx, loop in enumerate(mesh_cc.loops):
    uv_cc.data[loop_idx].uv = uv_data_cc[loop_idx]

mat_cc = bpy.data.materials.new(name="Mat_Slice_CC")
mat_cc.blend_method = 'BLEND'
bsdf_cc = mat_cc.node_tree.nodes.get("Principled BSDF")
tex_cc = mat_cc.node_tree.nodes.new(type="ShaderNodeTexImage")
tex_cc.image = bpy.data.images.load(r"{data['orig_cc_path']}")
mat_cc.node_tree.links.new(tex_cc.outputs['Color'], bsdf_cc.inputs['Base Color'])
if 'Emission Color' in bsdf_cc.inputs:
    mat_cc.node_tree.links.new(tex_cc.outputs['Color'], bsdf_cc.inputs['Emission Color'])
    bsdf_cc.inputs['Emission Strength'].default_value = 0.90

ramp_cc = mat_cc.node_tree.nodes.new(type="ShaderNodeValToRGB")
ramp_cc.color_ramp.elements[0].position = 0.02
ramp_cc.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 0.10)
ramp_cc.color_ramp.elements[1].position = 0.14
ramp_cc.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 0.95)
mat_cc.node_tree.links.new(tex_cc.outputs['Color'], ramp_cc.inputs['Fac'])
mat_cc.node_tree.links.new(ramp_cc.outputs['Alpha'], bsdf_cc.inputs['Alpha'])
obj_cc.data.materials.append(mat_cc)

# 4. Piano Verticale MLO (X = 0)
verts_mlo = [
    mathutils.Vector((0.0, -5.0, {z_inf_chest})),
    mathutils.Vector((0.0, -5.0, {z_sup_chest})),
    mathutils.Vector((0.0, {depth_mm + 4.0}, {z_sup_chest})),
    mathutils.Vector((0.0, {depth_mm + 4.0}, {z_inf_chest}))
]
mesh_mlo = bpy.data.meshes.new("mesh_slice_mlo")
mesh_mlo.from_pydata(verts_mlo, [], [(0, 1, 2, 3)])
uv_mlo = mesh_mlo.uv_layers.new(name="UVMap")

obj_mlo = bpy.data.objects.new("slice_mlo_plane", mesh_mlo)
bpy.context.collection.objects.link(obj_mlo)

uv_data_mlo = [
    ({u_chest_mlo}, {v_inf_mlo}),
    ({u_chest_mlo}, {v_sup_mlo}),
    ({u_nip_mlo}, {v_sup_mlo}),
    ({u_nip_mlo}, {v_inf_mlo})
]
for loop_idx, loop in enumerate(mesh_mlo.loops):
    uv_mlo.data[loop_idx].uv = uv_data_mlo[loop_idx]

mat_mlo = bpy.data.materials.new(name="Mat_Slice_MLO")
mat_mlo.blend_method = 'BLEND'
bsdf_mlo = mat_mlo.node_tree.nodes.get("Principled BSDF")
tex_mlo = mat_mlo.node_tree.nodes.new(type="ShaderNodeTexImage")
tex_mlo.image = bpy.data.images.load(r"{data['orig_mlo_path']}")
mat_mlo.node_tree.links.new(tex_mlo.outputs['Color'], bsdf_mlo.inputs['Base Color'])
if 'Emission Color' in bsdf_mlo.inputs:
    mat_mlo.node_tree.links.new(tex_mlo.outputs['Color'], bsdf_mlo.inputs['Emission Color'])
    bsdf_mlo.inputs['Emission Strength'].default_value = 0.90

ramp_mlo = mat_mlo.node_tree.nodes.new(type="ShaderNodeValToRGB")
ramp_mlo.color_ramp.elements[0].position = 0.02
ramp_mlo.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 0.10)
ramp_mlo.color_ramp.elements[1].position = 0.14
ramp_mlo.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 0.95)
mat_mlo.node_tree.links.new(tex_mlo.outputs['Color'], ramp_mlo.inputs['Fac'])
mat_mlo.node_tree.links.new(ramp_mlo.outputs['Alpha'], bsdf_mlo.inputs['Alpha'])
obj_mlo.data.materials.append(mat_mlo)

# 5. Punto di Focus e Telecamere Cliniche
target_focus = bpy.data.objects.new("Target_Focus", None)
target_focus.location = (0.0, {0.5 * depth_mm}, {0.05 * z_sup_chest})
bpy.context.collection.objects.link(target_focus)

def create_camera(name, loc, lens=52):
    cam_d = bpy.data.cameras.new(name)
    cam_d.lens = lens
    cam_o = bpy.data.objects.new(name, cam_d)
    cam_o.location = loc
    tt = cam_o.constraints.new(type='TRACK_TO')
    tt.target = target_focus
    tt.track_axis = 'TRACK_NEGATIVE_Z'
    tt.up_axis = 'UP_Y'
    bpy.context.collection.objects.link(cam_o)
    return cam_o

# Inquadratura Isometrica 3/4 (Riquadro 5)
cam_iso = create_camera("Camera_Isometric_Panel5", (320.0, 160.0, 95.0), lens=52)
# Inquadrature ortogonali cliniche (Riquadri 1, 2, 3, 6)
cam_mlo = create_camera("Camera_MLO_Panel1", (360.0, {0.5 * depth_mm}, 0.0), lens=55)
cam_cc = create_camera("Camera_CC_Panel2", (0.0, {0.5 * depth_mm}, 360.0), lens=55)
cam_sup = create_camera("Camera_Superior_Panel3", (0.0, 220.0, 240.0), lens=52)
cam_post = create_camera("Camera_Posterior_Panel6", (0.0, -280.0, 0.0), lens=55)

bpy.context.scene.camera = cam_iso

# 6. Luci da Studio Medico
sun1 = bpy.data.lights.new(name="Key_Light", type='SUN')
sun1.energy = 3.2
obj_sun1 = bpy.data.objects.new("Key_Light", sun1)
obj_sun1.rotation_euler = (0.8, 0.3, 0.9)
bpy.context.collection.objects.link(obj_sun1)

sun2 = bpy.data.lights.new(name="Fill_Light", type='SUN')
sun2.energy = 1.6
obj_sun2 = bpy.data.objects.new("Fill_Light", sun2)
obj_sun2.rotation_euler = (-0.6, -0.4, -0.9)
bpy.context.collection.objects.link(obj_sun2)

# 7. Configurazione Viewport su Material Preview per default
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'

# Salva scena .blend
bpy.ops.wm.save_as_mainfile(filepath=r"{os.path.abspath(blend_path)}")

# Renderizza preview isometrica ad alta risoluzione
scene = bpy.context.scene
scene.render.resolution_x = 1200
scene.render.resolution_y = 900
scene.render.filepath = "{preview_img_path}"
bpy.ops.render.render(write_still=True)
print("Scena Blender salvata e render generato con successo!")
"""
    tmp_file = os.path.join(os.path.dirname(blend_path), "_temp_build.py")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(script)

    import subprocess
    cmd = [BLENDER_EXE, "--background", "--python", tmp_file]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    if proc.returncode != 0:
        print("Errore Blender:", proc.stderr)
    else:
        print("Scena Blender generata con successo!")

def main():
    parser = argparse.ArgumentParser(description="Ricostruzione 3D Anatomica della Mammella da Mammografie CC ed MLO")
    parser.add_argument("--patient_id", type=str, default="P_00005", help="ID del paziente (es. P_00005, P_00001)")
    parser.add_argument("--data_dir", type=str, default="data", help="Percorso della cartella contenente i dati dei pazienti")
    parser.add_argument("--depth_mm", type=float, default=105.0, help="Profondita anatomica target in mm")
    args = parser.parse_args()

    patient_id = args.patient_id.strip()
    data_dir = args.data_dir
    depth_mm = args.depth_mm

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "output", patient_id)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"RICOSTRUZIONE 3D ANATOMICA ORGANICA A GOCCIA - Paziente {patient_id}")
    print("=" * 80)

    # Ricerca automatica delle lastre CC ed MLO
    laterality, cc_path, mlo_path = find_mammogram_pair(data_dir, patient_id)
    print(f"  Lateralià rilevata: Seno {'Destro (R)' if laterality == 'R' else 'Sinistro (L)'}")
    print(f"  CC : {cc_path}")
    print(f"  MLO: {mlo_path}")

    print("\n[1/4] Estrazione profili anatomici via ray-marching dal capezzolo...")
    data = extract_organic_profiles(cc_path, mlo_path, target_depth_mm=depth_mm, laterality=laterality)
    print(f"  Profondità anatomica: {data['target_depth_mm']:.1f} mm ({data['depth_px']} px)")

    print("\n[2/4] Costruzione mesh 3D anatomica a goccia fisiologica...")
    mesh, bounds_cc, bounds_mlo = build_teardrop_anatomical_mesh(data)
    w_mm = bounds_cc[1] - bounds_cc[0]
    h_mm = bounds_mlo[1] - bounds_mlo[0]
    d_mm = bounds_cc[2]
    print(f"  Mesh generata: {len(mesh.vertices)} vertici, {len(mesh.faces)} triangoli.")
    print(f"  Dimensioni: Larghezza {w_mm:.1f} mm, Altezza {h_mm:.1f} mm, Profondità {d_mm:.1f} mm")

    obj_path = os.path.join(out_dir, "breast_skin_surface.obj")
    glb_path = os.path.join(out_dir, "breast_skin_surface.glb")
    mesh.export(obj_path)
    mesh.export(glb_path)
    print(f"  File esportati: {obj_path}, {glb_path}")

    print("\n[3/4] Generazione Scena Blender con Aderenza Perfetta alle Lastre...")
    blend_path = os.path.join(out_dir, "breast_surface_scene.blend")
    generate_blender_scene(blend_path, mesh, data, bounds_cc, bounds_mlo)

    print("\n[4/4] Salvataggio metriche anatomiche...")
    metrics = {
        'patient_id': patient_id,
        'laterality': laterality,
        'depth_mm': float(d_mm),
        'width_mm': float(w_mm),
        'height_mm': float(h_mm),
        'volume_cm3': float(mesh.volume / 1000.0) if mesh.is_watertight else 0.0,
        'surface_area_cm2': float(mesh.area / 100.0),
        'mm_per_px': float(data['mm_per_px']),
        'vertex_count': len(mesh.vertices),
        'triangle_count': len(mesh.faces),
        'files': {
            'blend': os.path.abspath(blend_path),
            'obj': os.path.abspath(obj_path),
            'glb': os.path.abspath(glb_path),
            'preview_png': os.path.abspath(blend_path.replace('.blend', '_preview.png'))
        }
    }
    with open(os.path.join(out_dir, "surface_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("PROCESSO COMPLETATO CON SUCCESSO!")
    print(f"File di scena Blender aggiornato: {os.path.abspath(blend_path)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
