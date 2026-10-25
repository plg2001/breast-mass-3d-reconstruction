#!/usr/bin/env python3
import os
import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d

BLENDER_EXE = r"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"

def extract_breast_contours(cc_path, mlo_path, target_depth_mm=120.0):
    img_cc = cv2.imread(cc_path, cv2.IMREAD_GRAYSCALE)
    img_mlo = cv2.imread(mlo_path, cv2.IMREAD_GRAYSCALE)
    
    scale_cc = 600.0 / max(img_cc.shape)
    scale_mlo = 600.0 / max(img_mlo.shape)
    
    s_cc = cv2.resize(img_cc, (int(img_cc.shape[1] * scale_cc), int(img_cc.shape[0] * scale_cc)))
    s_mlo = cv2.resize(img_mlo, (int(img_mlo.shape[1] * scale_mlo), int(img_mlo.shape[0] * scale_mlo)))
    
    # Pulizia bordi
    for s in [s_cc, s_mlo]:
        s[:20, :] = 0; s[-20:, :] = 0; s[:, :20] = 0; s[:, -20:] = 0
        
    def get_clean_mask(small):
        _, b = cv2.threshold(small, 20, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8)

    m_cc = get_clean_mask(s_cc)
    m_mlo = get_clean_mask(s_mlo)
    
    # Nipple & Base detection in MLO
    cols_mlo = []
    for r in range(s_mlo.shape[0]):
        c = np.where(m_mlo[r, :] > 0)[0]
        if len(c) > 10:
            cols_mlo.append((r, c.min()))
    cols_mlo = np.array(cols_mlo)
    nip_mlo_idx = np.argmin(cols_mlo[:, 1])
    nip_r_mlo, nip_c_mlo = cols_mlo[nip_mlo_idx]
    
    # Nipple & Base detection in CC
    cols_cc = []
    for r in range(s_cc.shape[0]):
        c = np.where(m_cc[r, :] > 0)[0]
        if len(c) > 10:
            cols_cc.append((r, c.min()))
    cols_cc = np.array(cols_cc)
    nip_cc_idx = np.argmin(cols_cc[:, 1])
    nip_r_cc, nip_c_cc = cols_cc[nip_cc_idx]
    
    # Definisci la profondità del corpo mammario (dal capezzolo alla base retromammaria)
    # Evitando il prolungamento toracico posteriore eccessivo
    depth_px = 165 # profondita anatomica pura della mammella
    mm_per_px = target_depth_mm / float(depth_px)
    
    # Campiona profili da base a capezzolo (d=0 è base, d=depth_px è capezzolo)
    r_mins_cc, r_maxs_cc = [], []
    for d in range(depth_px + 1):
        col = nip_c_cc + (depth_px - d)
        col = int(np.clip(col, 0, s_cc.shape[1] - 1))
        rows = np.where(m_cc[:, col] > 0)[0]
        if len(rows) > 0:
            r_mins_cc.append(rows.min())
            r_maxs_cc.append(rows.max())
        else:
            r_mins_cc.append(nip_r_cc)
            r_maxs_cc.append(nip_r_cc)
            
    r_mins_mlo, r_maxs_mlo = [], []
    for d in range(depth_px + 1):
        col = nip_c_mlo + (depth_px - d)
        col = int(np.clip(col, 0, s_mlo.shape[1] - 1))
        rows = np.where(m_mlo[:, col] > 0)[0]
        if len(rows) > 0:
            # Limita l'estensione ascellare superiore a 1.25 volte il polo inferiore per forma anatomica a goccia
            inf_span = max(10, rows.max() - nip_r_mlo)
            sup_limit = nip_r_mlo - int(1.22 * inf_span)
            r_mins_mlo.append(max(rows.min(), sup_limit))
            r_maxs_mlo.append(rows.max())
        else:
            r_mins_mlo.append(nip_r_mlo)
            r_maxs_mlo.append(nip_r_mlo)
            
    return {
        'orig_cc_path': os.path.abspath(cc_path),
        'orig_mlo_path': os.path.abspath(mlo_path),
        's_cc_shape': s_cc.shape,
        's_mlo_shape': s_mlo.shape,
        'nip_r_cc': nip_r_cc, 'nip_c_cc': nip_c_cc,
        'nip_r_mlo': nip_r_mlo, 'nip_c_mlo': nip_c_mlo,
        'depth_px': depth_px,
        'mm_per_px': mm_per_px,
        'r_mins_cc': gaussian_filter1d(np.array(r_mins_cc, dtype=float), 3.0),
        'r_maxs_cc': gaussian_filter1d(np.array(r_maxs_cc, dtype=float), 3.0),
        'r_mins_mlo': gaussian_filter1d(np.array(r_mins_mlo, dtype=float), 3.0),
        'r_maxs_mlo': gaussian_filter1d(np.array(r_maxs_mlo, dtype=float), 3.0),
    }

def build_teardrop_breast_mesh(data, target_depth_mm=120.0, num_slices=85, num_radial=64):
    """
    Costruisce una mesh 3D anatomica a goccia, chiudendo morbidamente la base posteriore
    (eliminando completamente il fastidioso tubo toracico cilindrico).
    """
    mm_per_px = data['mm_per_px']
    t_samples = np.linspace(0.0, 1.0, num_slices)
    
    # Curva di chiusura posteriore morbida (lo spazio retromammario / cupola convessa posteriore)
    # Per t in [0, 0.16], scala a cupola sferica
    post_cap = np.where(t_samples < 0.16, np.sin((t_samples / 0.16) * (np.pi / 2.0)), 1.0)
    
    # Campionamento CC
    idx_cc = np.clip((t_samples * data['depth_px']).astype(int), 0, data['depth_px'])
    cc_min = data['r_mins_cc'][idx_cc]
    cc_max = data['r_maxs_cc'][idx_cc]
    X_lat = (data['nip_r_cc'] - cc_min) * mm_per_px * post_cap
    X_med = (data['nip_r_cc'] - cc_max) * mm_per_px * post_cap
    Xc_vals = 0.5 * (X_med + X_lat)
    W_vals = np.maximum(2.0, X_lat - X_med)
    
    # Campionamento MLO
    idx_mlo = np.clip((t_samples * data['depth_px']).astype(int), 0, data['depth_px'])
    mlo_min = data['r_mins_mlo'][idx_mlo]
    mlo_max = data['r_maxs_mlo'][idx_mlo]
    Z_sup = (data['nip_r_mlo'] - mlo_min) * mm_per_px * post_cap
    Z_inf = (data['nip_r_mlo'] - mlo_max) * mm_per_px * post_cap
    Zc_vals = 0.5 * (Z_sup + Z_inf)
    H_vals = np.maximum(2.0, Z_sup - Z_inf)
    
    Y_vals = t_samples * target_depth_mm
    
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
        
        # Curvatura posteriore dolce convessa a cupola per t < 0.16
        dome_offset = -6.0 * (1.0 - post_cap[i])
        
        for j in range(num_radial):
            x = xc + 0.5 * w * cos_sign[j]
            z = zc + 0.5 * h * sin_sign[j]
            vertices.append([x, y + dome_offset, z])
            
    # Centro posteriore
    chest_center_idx = len(vertices)
    vertices.append([Xc_vals[0], -6.0, Zc_vals[0]])
    
    # Apice capezzolo
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
    
    # Bounds dei piani
    bounds_cc = (float(np.min(X_med)), float(np.max(X_lat)), float(target_depth_mm))
    bounds_mlo = (float(np.min(Z_inf)), float(np.max(Z_sup)), float(target_depth_mm))
    
    return mesh, bounds_cc, bounds_mlo

def run_test():
    data = extract_breast_contours('data/P_00005/R-CC.jpg', 'data/P_00005/R-MLO.jpg', target_depth_mm=120.0)
    mesh, b_cc, b_mlo = build_teardrop_breast_mesh(data, target_depth_mm=120.0)
    print(f'Mesh bounds X: [{mesh.bounds[0][0]:.1f}, {mesh.bounds[1][0]:.1f}] mm (Width: {mesh.bounds[1][0]-mesh.bounds[0][0]:.1f} mm)')
    print(f'Mesh bounds Y: [{mesh.bounds[0][1]:.1f}, {mesh.bounds[1][1]:.1f}] mm (Depth: {mesh.bounds[1][1]-mesh.bounds[0][1]:.1f} mm)')
    print(f'Mesh bounds Z: [{mesh.bounds[0][2]:.1f}, {mesh.bounds[1][2]:.1f}] mm (Height: {mesh.bounds[1][2]-mesh.bounds[0][2]:.1f} mm)')
    
if __name__ == '__main__':
    run_test()
