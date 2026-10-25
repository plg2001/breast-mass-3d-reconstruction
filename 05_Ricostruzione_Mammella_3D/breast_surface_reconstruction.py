#!/usr/bin/env python3
"""
========================================================================================
Ricostruzione Tridimensionale dell'Inviluppo di Superficie della Mammella (Skin Envelope)
da Mammografia Bi-Planare (Viste CC ed MLO)
========================================================================================

Orientamento Anatomico nello Spazio 3D / Blender:
  +Y : Direzione Anteriore (dal torace Y=0 verso il capezzolo Y=target_depth_mm)
  +Z : Direzione Superiore (verso la testa / cavo ascellare)
  -Z : Direzione Inferiore (verso l'addome / solco sottomammario)
  +X : Direzione Laterale (verso il braccio)
  -X : Direzione Mediale (verso lo sterno)
"""

import os
import sys
import argparse
import json
import time
import subprocess
import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter1d

BLENDER_EXE = r"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"

def extract_view_profile(image_path, target_depth_mm=140.0):
    """
    Estrae il profilo della cute, la posizione del capezzolo e i limiti anatomici
    della mammella, eliminando artefatti di scansione.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossibile aprire: {image_path}")

    h_orig, w_orig = img.shape
    scale = 600.0 / max(h_orig, w_orig)
    small = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))

    # Pulizia artefatti scanner sui bordi superiore, inferiore e opposto alla parete toracica
    clean = small.copy()
    clean[:8, :] = 0; clean[-8:, :] = 0; clean[:, :8] = 0

    _, bin_img = cv2.threshold(clean, 25, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_idx).astype(np.uint8)

    w = small.shape[1]
    is_chest_right = mask[:, 3 * w // 4:].mean() > mask[:, :w // 4].mean()

    # Identificazione del capezzolo (apice colonna tessuto)
    cols_tissue = []
    for r in range(small.shape[0]):
        c = np.where(mask[r, :] > 0)[0]
        if len(c) > 10:
            cols_tissue.append((r, c.min() if is_chest_right else c.max()))
    cols_tissue = np.array(cols_tissue)

    if is_chest_right:
        nip_idx = np.argmin(cols_tissue[:, 1])
        nip_r, nip_c = cols_tissue[nip_idx]
        chest_c = w - 1
        depth_px = chest_c - nip_c
    else:
        nip_idx = np.argmax(cols_tissue[:, 1])
        nip_r, nip_c = cols_tissue[nip_idx]
        chest_c = 0
        depth_px = nip_c - chest_c

    mm_per_px = target_depth_mm / max(1, depth_px)

    # Estrazione profili min e max lungo la profondita torace -> capezzolo
    r_mins, r_maxs = [], []
    for d in range(depth_px + 1):
        col = (chest_c - d) if is_chest_right else (chest_c + d)
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) > 0:
            r_mins.append(rows.min())
            r_maxs.append(rows.max())
        else:
            r_mins.append(nip_r)
            r_maxs.append(nip_r)

    r_mins_smooth = gaussian_filter1d(np.array(r_mins, dtype=float), sigma=4.0)
    r_maxs_smooth = gaussian_filter1d(np.array(r_maxs, dtype=float), sigma=4.0)

    return {
        'orig_path': os.path.abspath(image_path),
        'orig_shape': (h_orig, w_orig),
        'small_shape': small.shape,
        'small_img': small,
        'scale': scale,
        'is_chest_right': is_chest_right,
        'chest_c': chest_c,
        'nip_r': nip_r,
        'nip_c': nip_c,
        'depth_px': depth_px,
        'mm_per_px': mm_per_px,
        'r_mins': r_mins_smooth,
        'r_maxs': r_maxs_smooth,
        'r_min_chest': r_mins_smooth[0],
        'r_max_chest': r_maxs_smooth[0]
    }

def build_anatomical_mesh(cc, mlo, target_depth_mm=140.0, num_slices=75, num_radial=64):
    """
    Costruisce la mesh 3D in coordinate anatomiche reali.
    """
    avg_depth_px = 0.5 * (cc['depth_px'] + mlo['depth_px'])
    mm_per_px = target_depth_mm / max(1.0, avg_depth_px)

    t_samples = np.linspace(0.0, 1.0, num_slices)

    # Vista CC (Asse X: Mediale - Laterale)
    idx_cc = np.clip((t_samples * cc['depth_px']).astype(int), 0, cc['depth_px'])
    cc_min = cc['r_mins'][idx_cc]
    cc_max = cc['r_maxs'][idx_cc]
    X_med = (cc_min - cc['nip_r']) * mm_per_px
    X_lat = (cc_max - cc['nip_r']) * mm_per_px
    Xc_vals = 0.5 * (X_med + X_lat)
    W_vals = np.maximum(2.0, X_lat - X_med)

    # Vista MLO (Asse Z: Supero - Inferiore)
    idx_mlo = np.clip((t_samples * mlo['depth_px']).astype(int), 0, mlo['depth_px'])
    mlo_min = mlo['r_mins'][idx_mlo]
    mlo_max = mlo['r_maxs'][idx_mlo]
    Z_sup = (mlo['nip_r'] - mlo_min) * mm_per_px
    Z_inf = (mlo['nip_r'] - mlo_max) * mm_per_px
    Zc_vals = 0.5 * (Z_sup + Z_inf)
    H_vals = np.maximum(2.0, Z_sup - Z_inf)

    # Asse Y: profondita torace -> capezzolo
    Y_vals = t_samples * target_depth_mm

    thetas = np.linspace(0, 2 * np.pi, num_radial, endpoint=False)
    cos_th = np.cos(thetas)
    sin_th = np.sin(thetas)
    p_exp = 2.1
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

        for j in range(num_radial):
            x = xc + 0.5 * w * cos_sign[j]
            z = zc + 0.5 * h * sin_sign[j]
            # Curvatura convessa naturale della gabbia toracica
            chest_curve = -10.0 * ((1.0 - t) ** 2.0) * ((cos_sign[j]) ** 2.0 + 0.5 * (sin_sign[j]) ** 2.0)
            vertices.append([x, y + chest_curve, z])

    chest_center_idx = len(vertices)
    vertices.append([Xc_vals[0], -10.0, Zc_vals[0]])

    nipple_tip_idx = len(vertices)
    vertices.append([Xc_vals[-1], target_depth_mm, Zc_vals[-1]])

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

    bounds_cc = (float(X_med[0]), float(X_lat[0]), float(target_depth_mm))
    bounds_mlo = (float(Z_inf[0]), float(Z_sup[0]), float(target_depth_mm))

    return mesh, mm_per_px, bounds_cc, bounds_mlo

def render_multiview_figure(mesh, cc, mlo, mm_per_px, output_png_path, laterality="R"):
    """
    Renderizza la figura multi-view a 6 viste.
    """
    fig = plt.figure(figsize=(18, 12), facecolor='#05070a')

    verts = mesh.vertices
    faces = mesh.faces

    # Calcolo colori ombreggiati
    normals = mesh.face_normals
    ld1 = np.array([0.6, 0.4, 0.7]) / np.linalg.norm([0.6, 0.4, 0.7])
    ld2 = np.array([-0.5, 0.3, 0.5]) / np.linalg.norm([-0.5, 0.3, 0.5])
    intensity = 0.65 * np.maximum(0.0, np.dot(normals, ld1)) + 0.35 * np.maximum(0.0, np.dot(normals, ld2))
    tot_light = np.clip(0.30 + 0.70 * intensity, 0.0, 1.0)
    face_colors = np.outer(tot_light, [0.84, 0.81, 0.78])

    lat_str = "destra" if laterality == "R" else "sinistra"

    def setup_ax(ax, title):
        ax.set_facecolor('#05070a')
        ax.set_title(title, color='#f0f6fc', fontsize=13, fontweight='medium', pad=10, loc='left')
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.line.set_color('#05070a')
        ax.yaxis.line.set_color('#05070a')
        ax.zaxis.line.set_color('#05070a')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # 1. Vista laterale MLO (vista lungo -X)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    p1 = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolors='none', shade=False)
    ax1.add_collection3d(p1)
    ax1.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax1.view_init(elev=0, azim=0)
    setup_ax(ax1, f"Vista laterale {lat_str}\n({laterality}-MLO)")

    # 2. Vista frontale CC (vista lungo -Y)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    p2 = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolors='none', shade=False)
    ax2.add_collection3d(p2)
    ax2.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax2.view_init(elev=0, azim=-90)
    setup_ax(ax2, f"Vista frontale\n({laterality}-CC)")

    # 3. Vista superiore (+Z) con gizmo coordinate
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    p3 = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolors='none', shade=False)
    ax3.add_collection3d(p3)
    ax3.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax3.view_init(elev=75, azim=-90)
    setup_ax(ax3, "Vista superiore")
    ax3.quiver(60, 20, 60, 30, 0, 0, color='#e63946', linewidth=2.5, arrow_length_ratio=0.2)
    ax3.quiver(60, 20, 60, 0, 30, 0, color='#457b9d', linewidth=2.5, arrow_length_ratio=0.2)
    ax3.quiver(60, 20, 60, 0, 0, 30, color='#2a9d8f', linewidth=2.5, arrow_length_ratio=0.2)
    ax3.text(95, 20, 60, 'X', color='#e63946', fontsize=10, fontweight='bold')
    ax3.text(60, 55, 60, 'Y', color='#457b9d', fontsize=10, fontweight='bold')
    ax3.text(60, 20, 95, 'Z', color='#2a9d8f', fontsize=10, fontweight='bold')

    # 4. Mesh 3D Wireframe
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    wire_colors = np.full((len(faces), 4), [0.12, 0.14, 0.18, 0.4])
    p4 = Poly3DCollection(verts[faces], facecolors=wire_colors, edgecolors='#d0d7de', linewidths=0.35, shade=False)
    ax4.add_collection3d(p4)
    ax4.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax4.view_init(elev=0, azim=0)
    setup_ax(ax4, f"Mesh 3D\n(vista laterale {lat_str})")

    # 5. Sezioni 2D (CC e MLO) sul modello 3D
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    transp_colors = np.full((len(faces), 4), [0.85, 0.85, 0.85, 0.15])
    p5 = Poly3DCollection(verts[faces], facecolors=transp_colors, edgecolors='#ffffff', linewidths=0.15, shade=False)
    ax5.add_collection3d(p5)

    # Piano MLO
    mlo_sub = mlo['small_img'][::6, ::6]
    h_m, w_m = mlo_sub.shape
    y_mlo_coords = np.linspace(0, mlo['depth_px'] * mm_per_px, w_m)
    z_mlo_coords = np.linspace(
        (mlo['nip_r'] - 0) * mm_per_px,
        (mlo['nip_r'] - mlo['small_img'].shape[0]) * mm_per_px,
        h_m
    )
    Y_grid_m, Z_grid_m = np.meshgrid(y_mlo_coords, z_mlo_coords)
    X_grid_m = np.zeros_like(Y_grid_m)
    colors_m = plt.cm.gray(mlo_sub.astype(float) / 255.0)
    colors_m[..., 3] = np.where(mlo_sub > 15, 0.85, 0.05)
    ax5.plot_surface(X_grid_m, Y_grid_m, Z_grid_m, rstride=1, cstride=1, facecolors=colors_m, shade=False)

    # Piano CC
    cc_sub = cc['small_img'][::6, ::6]
    h_c, w_c = cc_sub.shape
    y_cc_coords = np.linspace(0, cc['depth_px'] * mm_per_px, w_c)
    x_cc_coords = np.linspace(
        (0 - cc['nip_r']) * mm_per_px,
        (cc['small_img'].shape[0] - cc['nip_r']) * mm_per_px,
        h_c
    )
    X_grid_c, Y_grid_c = np.meshgrid(x_cc_coords, y_cc_coords, indexing='ij')
    Z_grid_c = np.zeros_like(X_grid_c)
    colors_c = plt.cm.gray(cc_sub.astype(float) / 255.0)
    colors_c[..., 3] = np.where(cc_sub > 15, 0.85, 0.05)
    ax5.plot_surface(X_grid_c, Y_grid_c, Z_grid_c, rstride=1, cstride=1, facecolors=colors_c, shade=False)

    ax5.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax5.view_init(elev=20, azim=-55)
    setup_ax(ax5, "Sezioni 2D (CC e MLO)\nsul modello 3D")

    # 6. Vista posteriore con barra metrica
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    p6 = Poly3DCollection(verts[faces], facecolors=face_colors, edgecolors='none', shade=False)
    ax6.add_collection3d(p6)
    ax6.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax6.view_init(elev=0, azim=180)
    setup_ax(ax6, "Vista posteriore")

    scale_y = np.min(verts[:, 1]) - 10
    scale_x = np.max(verts[:, 0]) - 20
    scale_z = np.min(verts[:, 2]) - 10
    for tick_val, label in [(0, '0'), (50, '5'), (100, '10'), (150, '15 cm')]:
        ax6.text(scale_x + tick_val - 150, scale_y, scale_z, label, color='#8b949e', fontsize=9)
    ax6.plot([scale_x - 150, scale_x], [scale_y, scale_y], [scale_z, scale_z], color='#8b949e', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def build_blender_scene(blend_path, mesh, cc, mlo, bounds_cc, bounds_mlo):
    """
    Costruisce la scena 3D direttamente in Blender con orientamento e materiali corretti.
    """
    if not os.path.exists(BLENDER_EXE):
        return False

    verts_list = mesh.vertices.tolist()
    faces_list = mesh.faces.tolist()

    x_med_chest, x_lat_chest, depth_mm = bounds_cc
    z_inf_chest, z_sup_chest, _ = bounds_mlo

    h_c, w_c = cc['small_shape']
    u_chest_cc = 1.0 if cc['is_chest_right'] else 0.0
    u_nip_cc = float(cc['nip_c']) / float(w_c)
    v_med_cc = 1.0 - (float(cc['r_min_chest']) / float(h_c))
    v_lat_cc = 1.0 - (float(cc['r_max_chest']) / float(h_c))

    h_m, w_m = mlo['small_shape']
    u_chest_mlo = 1.0 if mlo['is_chest_right'] else 0.0
    u_nip_mlo = float(mlo['nip_c']) / float(w_m)
    v_sup_mlo = 1.0 - (float(mlo['r_min_chest']) / float(h_m))
    v_inf_mlo = 1.0 - (float(mlo['r_max_chest']) / float(h_m))

    script = f"""
import bpy
import mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Creazione Mesh Cutanea 3D Liscia
mesh_data = bpy.data.meshes.new("mesh_breast_skin")
mesh_data.from_pydata({verts_list}, [], {faces_list})
mesh_data.polygons.foreach_set('use_smooth', [True] * len(mesh_data.polygons))
mesh_data.update()

skin_obj = bpy.data.objects.new("breast_skin_surface", mesh_data)
bpy.context.collection.objects.link(skin_obj)

# Materiale Alabastro Cutaneo Semitrasparente
mat_skin = bpy.data.materials.new(name="Mat_Breast_Skin")
mat_skin.diffuse_color = (0.86, 0.82, 0.79, 0.38)
bsdf = mat_skin.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.86, 0.82, 0.79, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.35
    if 'Alpha' in bsdf.inputs:
        bsdf.inputs['Alpha'].default_value = 0.38
skin_obj.data.materials.append(mat_skin)

# 2. Piano Orizzontale CC (Z = 0)
verts_cc = [
    mathutils.Vector(({x_med_chest}, 0.0, 0.0)),
    mathutils.Vector(({x_lat_chest}, 0.0, 0.0)),
    mathutils.Vector(({x_lat_chest}, {depth_mm}, 0.0)),
    mathutils.Vector(({x_med_chest}, {depth_mm}, 0.0))
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
bsdf_cc = mat_cc.node_tree.nodes.get("Principled BSDF")
tex_cc = mat_cc.node_tree.nodes.new(type="ShaderNodeTexImage")
tex_cc.image = bpy.data.images.load(r"{cc['orig_path']}")
mat_cc.node_tree.links.new(tex_cc.outputs['Color'], bsdf_cc.inputs['Base Color'])
if 'Emission Color' in bsdf_cc.inputs:
    mat_cc.node_tree.links.new(tex_cc.outputs['Color'], bsdf_cc.inputs['Emission Color'])
    bsdf_cc.inputs['Emission Strength'].default_value = 0.7
obj_cc.data.materials.append(mat_cc)

# 3. Piano Verticale MLO (X = 0)
verts_mlo = [
    mathutils.Vector((0.0, 0.0, {z_inf_chest})),
    mathutils.Vector((0.0, 0.0, {z_sup_chest})),
    mathutils.Vector((0.0, {depth_mm}, {z_sup_chest})),
    mathutils.Vector((0.0, {depth_mm}, {z_inf_chest}))
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
bsdf_mlo = mat_mlo.node_tree.nodes.get("Principled BSDF")
tex_mlo = mat_mlo.node_tree.nodes.new(type="ShaderNodeTexImage")
tex_mlo.image = bpy.data.images.load(r"{mlo['orig_path']}")
mat_mlo.node_tree.links.new(tex_mlo.outputs['Color'], bsdf_mlo.inputs['Base Color'])
if 'Emission Color' in bsdf_mlo.inputs:
    mat_mlo.node_tree.links.new(tex_mlo.outputs['Color'], bsdf_mlo.inputs['Emission Color'])
    bsdf_mlo.inputs['Emission Strength'].default_value = 0.7
obj_mlo.data.materials.append(mat_mlo)

# 4. Luci Studio
key_light = bpy.data.lights.new(name="Key_Light", type='SUN')
key_light.energy = 3.2
key_obj = bpy.data.objects.new("Key_Light", key_light)
key_obj.rotation_euler = (0.75, 0.35, 0.85)
bpy.context.collection.objects.link(key_obj)

fill_light = bpy.data.lights.new(name="Fill_Light", type='SUN')
fill_light.energy = 1.4
fill_obj = bpy.data.objects.new("Fill_Light", fill_light)
fill_obj.rotation_euler = (-0.6, -0.4, -0.9)
bpy.context.collection.objects.link(fill_obj)

# 5. Camera 3D fronto-laterale
cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
cam_obj.location = (260, 310, 180)
cam_obj.rotation_euler = (1.18, 0.0, 2.45)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Imposta Viewport su Material Preview per visualizzare subito trasparenza e texture
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'

bpy.ops.wm.save_as_mainfile(filepath=r"{os.path.abspath(blend_path)}")
"""
    tmp_py = os.path.join(os.path.dirname(blend_path), "_temp_build_scene.py")
    with open(tmp_py, "w", encoding="utf-8") as f:
        f.write(script)

    cmd = [BLENDER_EXE, "--background", "--python", tmp_py]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(tmp_py):
        os.remove(tmp_py)

    return res.returncode == 0

def process_patient(patient_id="P_00005", data_root="data", output_base="05_Ricostruzione_Mammella_3D/output",
                    target_depth_mm=140.0, cc_path=None, mlo_path=None):
    """
    Esegue la pipeline di ricostruzione per il paziente.
    """
    t0 = time.time()
    patient_dir = os.path.join(data_root, patient_id)
    out_dir = os.path.join(output_base, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    if cc_path is None or mlo_path is None:
        r_cc = os.path.join(patient_dir, "R-CC.jpg")
        r_mlo = os.path.join(patient_dir, "R-MLO.jpg")
        l_cc = os.path.join(patient_dir, "L-CC.jpg")
        l_mlo = os.path.join(patient_dir, "L-MLO.jpg")

        if os.path.exists(r_cc) and os.path.exists(r_mlo):
            cc_path, mlo_path = r_cc, r_mlo
            laterality = "R"
        elif os.path.exists(l_cc) and os.path.exists(l_mlo):
            cc_path, mlo_path = l_cc, l_mlo
            laterality = "L"
        else:
            raise FileNotFoundError(f"Impossibile trovare una coppia CC+MLO in {patient_dir}")
    else:
        laterality = "R" if "R-" in os.path.basename(cc_path) else "L"

    print(f"\n=======================================================")
    print(f" Ricostruzione Superficie 3D Mammella: {patient_id} ({laterality})")
    print(f" CC:  {cc_path}")
    print(f" MLO: {mlo_path}")
    print(f"=======================================================")

    print("[1/5] Estrazione contorni cutanei e landmark anatomici...")
    cc = extract_view_profile(cc_path, target_depth_mm=target_depth_mm)
    mlo = extract_view_profile(mlo_path, target_depth_mm=target_depth_mm)

    print("[2/5] Generazione mesh 3D quad-ellittica continua...")
    mesh, mm_per_px, bounds_cc, bounds_mlo = build_anatomical_mesh(cc, mlo, target_depth_mm=target_depth_mm)

    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    volume_cm3 = mesh.volume / 1000.0 if mesh.is_volume else 0.0
    area_cm2 = mesh.area / 100.0

    print(f"      Dimensioni Bounding Box: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
    print(f"      Volume: {volume_cm3:.1f} cm3, Area: {area_cm2:.1f} cm2")
    print(f"      Watertight: {mesh.is_watertight} (Vertici: {len(mesh.vertices)}, Facce: {len(mesh.faces)})")

    print("[3/5] Esportazione file 3D (.obj, .glb, .blend)...")
    obj_path = os.path.join(out_dir, "breast_skin_surface.obj")
    glb_path = os.path.join(out_dir, "breast_skin_surface.glb")
    blend_path = os.path.join(out_dir, "breast_surface_scene.blend")

    mesh.export(obj_path)
    mesh.export(glb_path)

    has_blender = build_blender_scene(blend_path, mesh, cc, mlo, bounds_cc, bounds_mlo)
    if has_blender:
        print(f"      File nativo Blender generato: {blend_path}")

    print("[4/5] Generazione figura multi-view a 6 viste...")
    figure_png_path = os.path.join(out_dir, f"{patient_id}_breast_surface_multiview.png")
    render_multiview_figure(mesh, cc, mlo, mm_per_px, figure_png_path, laterality=laterality)

    elapsed = round(time.time() - t0, 2)
    metrics = {
        'patient_id': patient_id,
        'laterality': laterality,
        'depth_mm': target_depth_mm,
        'width_mm': float(round(dims[0], 2)),
        'height_mm': float(round(dims[2], 2)),
        'volume_cm3': float(round(volume_cm3, 2)),
        'surface_area_cm2': float(round(area_cm2, 2)),
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'watertight': bool(mesh.is_watertight),
        'files': {
            'obj': obj_path,
            'glb': glb_path,
            'blend': blend_path if has_blender else None,
            'figure_png': figure_png_path
        },
        'elapsed_seconds': elapsed
    }
    metrics_path = os.path.join(out_dir, "surface_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"[5/5] Completato con successo in {elapsed} secondi!\n")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Ricostruzione Superficie 3D della Mammella da Mammografie Bi-Planari")
    parser.add_argument('--patient_id', type=str, default='P_00005', help='ID Paziente (es. P_00005)')
    parser.add_argument('--data_root', type=str, default='data', help='Percorso cartella dati')
    parser.add_argument('--output_dir', type=str, default='05_Ricostruzione_Mammella_3D/output', help='Cartella di output')
    parser.add_argument('--cc_path', type=str, default=None, help='Percorso diretto file CC')
    parser.add_argument('--mlo_path', type=str, default=None, help='Percorso diretto file MLO')
    parser.add_argument('--depth_mm', type=float, default=140.0, help='Profondita torace-capezzolo in mm')

    args = parser.parse_args()
    process_patient(
        patient_id=args.patient_id,
        data_root=args.data_root,
        output_base=args.output_dir,
        target_depth_mm=args.depth_mm,
        cc_path=args.cc_path,
        mlo_path=args.mlo_path
    )

if __name__ == '__main__':
    main()
