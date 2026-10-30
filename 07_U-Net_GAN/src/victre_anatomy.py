"""
VICTRE Anatomical Extension Module
=================================
Porting and adaptation of the mathematical principles and procedural anatomical models
from FDA VICTRE (In Silico Breast Phantom - DIDSR / Christian Graff et al. 2016, IEEE TMI)
and University of Pennsylvania (Bakic et al., Med Phys) to patient-specific reconstructions:

1. Lactiferous Ductal Tree (duct.cxx / duct.hxx):
   - Sub-areolar lactiferous sinuses (distributed across an areolar disc)
   - Centroid-guided curved branching traversing all four breast quadrants (UOQ, UIQ, LOQ, LIQ)
   - Primary lactiferous trunks (r ~ 0.70 -> 0.55 mm) radiating into parenchymal lobes
   - Secondary bifurcations (r ~ 0.48 -> 0.35 mm) extending into deep glandular sectors
   - Tertiary ductules (r ~ 0.30 -> 0.22 mm) terminating in TDLUs (Terminal Ductal Lobular Units)
   - Organic curved branch geometry permeating the entire glandular parenchymal canopy
   - Confinement within the patient's individual fibroglandular boundary

2. Cooper's Ligaments / Retinacula Cutis (cooper.hxx):
   - Curvilinear fibrous suspensory retinacular bands anchoring parenchymal crests (Duret's crests)
     to the dermal skin envelope across the premammary fat space, with subdermal rootlet insertions

3. Subcutaneous Adipose Tissue / Premammary Fat Mantle (compartment.cxx):
   - Volumetric adipose layer conforming to the space between skin and glandular core
   - Lobular adipose texture variations conforming to the patient's breast geometry

4. Pectoralis Major Muscle Plate:
   - Anatomical posterior chest wall boundary and superior axillary slope (MLO projection)
   - Continuous, smoothly contoured muscular sheet without sharp planar bounding cuts
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
import trimesh
from skimage.measure import marching_cubes


def create_curved_tube(p1, p2, r_start, r_end, num_subsegs=2, curvature=0.8, sections=7):
    """
    Crea un ramo tubolare curvo e rastremato tra p1 e p2 per un aspetto biologico naturale.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    v = p2 - p1
    length = float(np.linalg.norm(v))
    if length < 0.6:
        return []

    ortho = np.array([-v[1], v[0], 0.0])
    ortho_len = np.linalg.norm(ortho)
    if ortho_len < 1e-4:
        ortho = np.array([1.0, 0.0, 0.0])
    else:
        ortho = ortho / ortho_len
    ortho = ortho * curvature

    t_vals = np.linspace(0.0, 1.0, num_subsegs + 1)
    pts = []
    for t in t_vals:
        sag = 4.0 * t * (1.0 - t)
        pt = (1.0 - t) * p1 + t * p2 + sag * ortho
        pts.append(pt)

    subsegs = []
    z_axis = np.array([0.0, 0.0, 1.0])

    for i in range(num_subsegs):
        pt_a = pts[i]
        pt_b = pts[i + 1]
        v_sub = pt_b - pt_a
        sub_len = float(np.linalg.norm(v_sub))
        if sub_len < 1e-4:
            continue

        r = float((1.0 - t_vals[i]) * r_start + t_vals[i] * r_end)
        cyl = trimesh.creation.cylinder(radius=r, height=sub_len, sections=sections)

        v_sub_norm = v_sub / sub_len
        axis = np.cross(z_axis, v_sub_norm)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            rot = np.eye(4)
            if v_sub_norm[2] < 0:
                rot[0, 0] = -1.0
                rot[2, 2] = -1.0
        else:
            ang = np.arccos(np.clip(np.dot(z_axis, v_sub_norm), -1.0, 1.0))
            rot = trimesh.transformations.rotation_matrix(ang, axis)

        trans = trimesh.transformations.translation_matrix((pt_a + pt_b) * 0.5)
        cyl.apply_transform(trans @ rot)
        subsegs.append(cyl)

    return subsegs


def get_parenchyma_point(kinematics, t_ratio, theta, rho=0.55):
    """
    Calcola le coordinate 3D all'interno del parenchima fibroglandolare del paziente
    a profondita t_ratio, angolo theta e frazione radiale rho rispetto al baricentro curvo.
    """
    pnl = kinematics.pnl_mm
    slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
    _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)
    
    a = max(2.5, 0.5 * W * 0.78)
    b = max(2.5, 0.5 * H * 0.78)
    
    # Nel polo superiore profondo, rispetta il margine del muscolo pettorale
    if t_ratio < 0.45 and np.sin(theta) > 0.35:
        b *= 0.72
        
    px = Xc + rho * a * np.cos(theta)
    py = Zc + rho * b * np.sin(theta)
    pz = t_ratio * pnl
    return np.array([px, py, pz])


def generate_victre_ductal_tree(kinematics, num_lobes=12, rng_seed=42):
    """
    Genera l'albero galattoforo completo secondo FDA VICTRE (duct.cxx).
    L'albero pervade la chioma parenchimale in tutti i quadranti anatomici
    dal complesso areola-capezzolo fino alla profondita retromammaria.
    """
    rng = np.random.RandomState(rng_seed)
    pnl = kinematics.pnl_mm
    _, _, _, Xc_apex, Zc_apex = kinematics.get_slice_parameters(kinematics.num_depth_slices - 1)
    
    segments = []
    
    for i in range(num_lobes):
        theta = 2.0 * np.pi * (i + rng.uniform(-0.10, 0.10)) / num_lobes
        
        # 1. Origine sul disco sub-areolare
        r_areola = rng.uniform(2.5, 5.0)
        p_origin = np.array([
            Xc_apex + r_areola * np.cos(theta),
            Zc_apex + r_areola * np.sin(theta),
            pnl - rng.uniform(2.0, 3.5)
        ])
        
        # 2. Tronco Primario: dal capezzolo alla profondità antero-media (t ~ 0.70)
        t_trunk = rng.uniform(0.68, 0.74)
        rho_trunk = rng.uniform(0.40, 0.65)
        p_trunk_end = get_parenchyma_point(kinematics, t_trunk, theta, rho=rho_trunk)
        
        segs0 = create_curved_tube(p_origin, p_trunk_end, r_start=0.70, r_end=0.52, 
                                   num_subsegs=2, curvature=rng.uniform(-0.8, 0.8), sections=7)
        segments.extend(segs0)
        
        # 3. Ramificazioni Secondarie: entrano nel corpo del lobo (t ~ 0.46 - 0.52)
        for b in [-1, 1]:
            d_theta_sec = b * rng.uniform(0.22, 0.38)
            theta_sec = theta + d_theta_sec
            t_sec = rng.uniform(0.46, 0.54)
            rho_sec = np.clip(rho_trunk + b * rng.uniform(0.08, 0.20), 0.30, 0.75)
            p_sec_end = get_parenchyma_point(kinematics, t_sec, theta_sec, rho=rho_sec)
            
            segs1 = create_curved_tube(p_trunk_end, p_sec_end, r_start=0.48, r_end=0.34,
                                       num_subsegs=2, curvature=rng.uniform(-0.6, 0.6), sections=6)
            segments.extend(segs1)
            
            # 4. Ramificazioni Terziarie e TDLU: penetrano nel parenchima profondo (t ~ 0.26 - 0.34)
            for c in [-1, 1]:
                d_theta_tert = c * rng.uniform(0.18, 0.32)
                theta_tert = theta_sec + d_theta_tert
                t_tert = rng.uniform(0.26, 0.36)
                rho_tert = np.clip(rho_sec + c * rng.uniform(0.06, 0.16), 0.25, 0.78)
                p_tert_end = get_parenchyma_point(kinematics, t_tert, theta_tert, rho=rho_tert)
                
                segs2 = create_curved_tube(p_sec_end, p_tert_end, r_start=0.32, r_end=0.22,
                                           num_subsegs=2, curvature=rng.uniform(-0.5, 0.5), sections=6)
                segments.extend(segs2)
                
                # Terminal Ductal Lobular Unit (TDLU)
                tdlu = trimesh.creation.icosphere(subdivisions=1, radius=0.55)
                tdlu.apply_translation(p_tert_end)
                segments.append(tdlu)

    tree_mesh = trimesh.util.concatenate(segments)
    duct_color = np.array([245, 248, 252, 235], dtype=np.uint8)
    tree_mesh.visual.vertex_colors = np.tile(duct_color, (len(tree_mesh.vertices), 1))
    return tree_mesh


def generate_cooper_ligaments(tissue_mesh, skin_mesh, kinematics, num_ligaments=26, rng_seed=42):
    """
    Genera i Legamenti di Cooper (retinacula cutis) secondo FDA VICTRE (cooper.hxx).
    Campiona le creste parenchimali nella fascia equatoriale/media (30% - 72% PNL)
    ed estende sottili bande connettivali arcuate con radichette subdermiche di inserzione.
    """
    pnl = kinematics.pnl_mm
    verts = tissue_mesh.vertices
    
    z_mask = (verts[:, 2] > 0.30 * pnl) & (verts[:, 2] < 0.72 * pnl)
    cand_verts = verts[z_mask]
    
    if len(cand_verts) == 0:
        return None
        
    angles = np.arctan2(cand_verts[:, 1], cand_verts[:, 0])
    num_bins = num_ligaments
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    
    crest_pts = []
    for b in range(num_bins):
        in_bin = (angles >= bins[b]) & (angles < bins[b+1])
        if np.any(in_bin):
            bin_v = cand_verts[in_bin]
            rad_sq = bin_v[:, 0]**2 + bin_v[:, 1]**2
            crest_pts.append(bin_v[np.argmax(rad_sq)])
            
    if len(crest_pts) == 0:
        return None
        
    crest_pts = np.array(crest_pts)
    skin_tree = cKDTree(skin_mesh.vertices)
    dists, indices = skin_tree.query(crest_pts)
    
    tubes = []
    rng = np.random.RandomState(rng_seed)
    
    for p_crest, dist, s_idx in zip(crest_pts, dists, indices):
        if 4.5 < dist < 22.0:
            p_skin = skin_mesh.vertices[s_idx]
            # Tronco principale del ligamento
            curved_segs = create_curved_tube(p_crest, p_skin, r_start=0.28, r_end=0.20,
                                             num_subsegs=2, curvature=0.6, sections=6)
            tubes.extend(curved_segs)
            
            # Piccole radichette di inserzione subdermica (retinacula cutis)
            v_dir = p_skin - p_crest
            v_len = np.linalg.norm(v_dir)
            if v_len > 6.0:
                p_mid = p_crest + v_dir * 0.75
                for s_sign in [-1, 1]:
                    offset = np.array([s_sign * rng.uniform(1.2, 2.5), rng.uniform(-1.0, 1.0), 0.0])
                    p_root = p_skin + offset
                    # Proietta alla cute
                    _, r_idx = skin_tree.query(p_root)
                    p_root_skin = skin_mesh.vertices[r_idx]
                    root_segs = create_curved_tube(p_mid, p_root_skin, r_start=0.18, r_end=0.12,
                                                   num_subsegs=1, curvature=0.0, sections=5)
                    tubes.extend(root_segs)
                
    if not tubes:
        return None
        
    cooper_mesh = trimesh.util.concatenate(tubes)
    cooper_color = np.array([225, 230, 240, 190], dtype=np.uint8)
    cooper_mesh.visual.vertex_colors = np.tile(cooper_color, (len(cooper_mesh.vertices), 1))
    return cooper_mesh


def generate_subcutaneous_fat(kinematics, skin_mesh):
    """
    Genera il mantello adiposo sottocutaneo / premammario (FDA VICTRE compartment.cxx).
    Occupa lo spazio anatomico tra la cute e il parenchima interno.
    """
    min_b, max_b = skin_mesh.bounds
    Nx, Ny, Nz = 60, 75, 60
    xs = np.linspace(min_b[0], max_b[0], Nx)
    ys = np.linspace(min_b[1], max_b[1], Ny)
    zs = np.linspace(0.0, kinematics.pnl_mm, Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    fat_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    for k, z_val in enumerate(zs):
        t_ratio = z_val / kinematics.pnl_mm
        slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
        _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)
        
        a_skin = max(1.5, 0.5 * W)
        b_skin = max(1.5, 0.5 * H)
        p = 2.15 - 0.15 * t_ratio
        
        dx = np.abs(X[:, :, k] - Xc) / a_skin
        dy = np.abs(Y[:, :, k] - Zc) / b_skin
        dist_se = (dx ** p + dy ** p) ** (1.0 / p)
        
        # Fascia adiposa compresa tra 0.86 e 0.985
        in_shell = np.exp(-((dist_se - 0.92) / 0.055)**2)
        lobules = 0.88 + 0.12 * np.sin(0.35 * X[:, :, k]) * np.cos(0.35 * Y[:, :, k])
        nipple_falloff = np.clip((0.97 - t_ratio) / 0.04, 0.0, 1.0) if t_ratio > 0.93 else 1.0
        
        fat_grid[:, :, k] = in_shell * lobules * nipple_falloff

    spacing = (
        (max_b[0] - min_b[0]) / (Nx - 1),
        (max_b[1] - min_b[1]) / (Ny - 1),
        kinematics.pnl_mm / (Nz - 1)
    )
    smooth_fat = ndimage.gaussian_filter(fat_grid, sigma=0.45)
    verts, faces, normals, _ = marching_cubes(smooth_fat, level=0.42, spacing=spacing)
    verts[:, 0] += min_b[0]
    verts[:, 1] += min_b[1]
    
    fat_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    fat_color = np.array([245, 235, 175, 70], dtype=np.uint8)
    fat_mesh.visual.vertex_colors = np.tile(fat_color, (len(fat_mesh.vertices), 1))
    return fat_mesh


def generate_pectoral_muscle(kinematics, skin_mesh):
    """
    Genera la placca muscolare del Grande Pettorale (Pectoralis Major)
    come lamina muscolare biologica continua e rastremata lungo la parete toracica
    e il margine supero-posteriore, con raccordi anatomici morbidi.
    """
    min_b, max_b = skin_mesh.bounds
    Nx, Ny, Nz = 60, 75, 45
    xs = np.linspace(min_b[0], max_b[0], Nx)
    ys = np.linspace(min_b[1], max_b[1], Ny)
    max_z_pec = min(28.0, kinematics.pnl_mm * 0.35)
    zs = np.linspace(0.0, max_z_pec, Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    pec_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    for k, z_val in enumerate(zs):
        t_ratio = z_val / kinematics.pnl_mm
        slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
        _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)
        
        a_skin = max(1.5, 0.5 * W * 0.95)
        b_skin = max(1.5, 0.5 * H * 0.95)
        p = 2.15 - 0.15 * t_ratio
        
        dx = np.abs(X[:, :, k] - Xc) / a_skin
        dy = np.abs(Y[:, :, k] - Zc) / b_skin
        dist_se = (dx ** p + dy ** p) ** (1.0 / p)
        
        in_bound = np.clip((1.0 - dist_se) / 0.12, 0.0, 1.0)
        v = np.clip(((Y[:, :, k] - Zc) / b_skin + 1.0) * 0.5, 0.0, 1.0)
        
        v_pec = 0.56 + 0.38 * (t_ratio / 0.35) if t_ratio < 0.35 else 1.5
        pec_mask = np.clip((v - v_pec) / 0.10, 0.0, 1.0)
        
        z_falloff = np.clip((max_z_pec - z_val) / 7.0, 0.0, 1.0) * np.clip((z_val + 0.8) / 2.8, 0.0, 1.0)
        base_plate = np.clip((4.5 - z_val) / 2.2, 0.0, 1.0) * np.clip((v - 0.18) / 0.25, 0.0, 1.0)
        
        pec_grid[:, :, k] = in_bound * np.maximum(pec_mask * z_falloff, base_plate)

    spacing = (
        (max_b[0] - min_b[0]) / (Nx - 1),
        (max_b[1] - min_b[1]) / (Ny - 1),
        max_z_pec / (Nz - 1)
    )
    smooth_pec = ndimage.gaussian_filter(pec_grid, sigma=0.80)
    verts, faces, normals, _ = marching_cubes(smooth_pec, level=0.34, spacing=spacing)
    verts[:, 0] += min_b[0]
    verts[:, 1] += min_b[1]
    
    pec_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    pec_color = np.array([175, 45, 45, 215], dtype=np.uint8)
    pec_mesh.visual.vertex_colors = np.tile(pec_color, (len(pec_mesh.vertices), 1))
    return pec_mesh
