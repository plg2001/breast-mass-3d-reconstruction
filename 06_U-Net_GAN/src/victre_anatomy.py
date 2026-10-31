"""
VICTRE Anatomical Extension Module - Hyper-Realistic Edition
===========================================================
Porting and adaptation of the mathematical principles and procedural anatomical models
from FDA VICTRE (In Silico Breast Phantom - DIDSR / Christian Graff et al. 2016, IEEE TMI)
and University of Pennsylvania (Bakic et al., Med Phys) to patient-specific reconstructions:

1. Lactiferous Ductal Tree (duct.cxx / duct.hxx):
   - Sub-areolar lactiferous sinuses (distributed across an areolar disc)
   - Centroid-guided curved branching traversing all four breast quadrants (UOQ, UIQ, LOQ, LIQ)
   - Primary trunks (r ~ 0.70 -> 0.52 mm), secondary branches (r ~ 0.48 -> 0.34 mm),
     tertiary ductules (r ~ 0.32 -> 0.22 mm), terminating in TDLUs (Terminal Ductal Lobular Units)
   - Organic curved branch geometry permeating the entire glandular parenchymal canopy

2. Superficial Venous Plexus (Circulus Venosus Halleri):
   - Subcutaneous anastomotic venous ring around areola (Haller's circle)
   - Branching subcutaneous veins draining medially, laterally, and superiorly

3. Deep Glandular Arterial Network:
   - Lateral thoracic and internal mammary perforating branches supplying parenchymal lobules

4. Nipple-Areola Complex (NAC):
   - Projecting nipple papilla with lactiferous orifices
   - Pigmented areolar disc and circumferential Montgomery tubercles

5. Cooper's Ligaments / Retinacula Cutis (cooper.hxx):
   - Curvilinear fibrous suspensory retinacular bands anchoring parenchymal crests (Duret's crests)
     to the dermal skin envelope across the premammary fat space, with subdermal rootlets

6. Intramammary Lymph Nodes:
   - Reniform (bean-shaped) lymph nodes located in the Upper Outer Quadrant (axillary pathway)

7. Subcutaneous Adipose Tissue / Premammary Fat Mantle (compartment.cxx):
   - Volumetric adipose layer conforming to the space between skin and glandular core

8. Retromammary Adipose Cushion & Space (Spazio di Chassaignac):
   - Adipose sliding plane between posterior parenchyma and deep pectoral fascia

9. Pectoralis Major Muscle Plate:
   - Anatomical posterior chest wall boundary and superior axillary slope (MLO projection)
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree
import trimesh
from skimage.measure import marching_cubes


def create_curved_tube(p1, p2, r_start, r_end, num_subsegs=2, curvature=0.7, sections=6):
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


def clamp_mesh_inside_skin(mesh, skin_mesh, min_inward_depth=3.0):
    """
    Garantisce matematicamente che NESSUN vertice di 'mesh' sporga all'esterno della cute.
    Se un vertice si trova all'esterno della cute o a una profondità inferiore a 'min_inward_depth',
    viene proiettato verso l'interno lungo la normale della superficie cutanea più vicina.
    """
    if mesh is None or len(mesh.vertices) == 0 or skin_mesh is None:
        return mesh

    skin_kdt = cKDTree(skin_mesh.vertices)
    _, indices = skin_kdt.query(mesh.vertices)

    skin_pts = skin_mesh.vertices[indices]
    skin_norms = np.array(skin_mesh.vertex_normals[indices], dtype=np.float64)
    norms = np.linalg.norm(skin_norms, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    skin_norms = skin_norms / norms

    vecs = mesh.vertices - skin_pts
    outward_dist = np.sum(vecs * skin_norms, axis=1)

    violations = outward_dist > -min_inward_depth
    if np.any(violations):
        mesh.vertices[violations] = skin_pts[violations] - min_inward_depth * skin_norms[violations]

    return mesh


def get_parenchyma_point(kinematics, t_ratio, theta, rho=0.55):
    """
    Calcola le coordinate 3D all'interno del parenchima fibroglandolare del paziente
    a profondità t_ratio, angolo theta e frazione radiale rho rispetto al baricentro curvo.
    """
    pnl = kinematics.pnl_mm
    slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
    _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)
    
    a = max(2.5, 0.5 * W * 0.78)
    b = max(2.5, 0.5 * H * 0.78)
    
    if t_ratio < 0.45 and np.sin(theta) > 0.35:
        b *= 0.72
        
    px = Xc + rho * a * np.cos(theta)
    py = Zc + rho * b * np.sin(theta)
    pz = t_ratio * pnl
    return np.array([px, py, pz])


def get_subcutaneous_point(kinematics, t_ratio, theta, rho=0.93):
    """
    Calcola le coordinate 3D sul mantello sottocutaneo (nello strato adiposo, 1-2 mm sotto la cute).
    """
    pnl = kinematics.pnl_mm
    slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
    _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)
    
    a = max(2.5, 0.5 * W * rho)
    b = max(2.5, 0.5 * H * rho)
    p = 2.15 - 0.15 * t_ratio
    
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    px = Xc + a * np.sign(cos_th) * (np.abs(cos_th) ** (2.0 / p))
    py = Zc + b * np.sign(sin_th) * (np.abs(sin_th) ** (2.0 / p))
    pz = t_ratio * pnl
    return np.array([px, py, pz])


def sample_neural_density(volume_3d, kinematics, pt):
    """
    Campiona la densità neurale [0, 1] predetta dalla rete neurale 3D U-Net-GAN
    per una coordinata spaziale reale 3D pt = (x, y, z) in mm.
    """
    if volume_3d is None:
        return 0.50

    px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
    pnl = kinematics.pnl_mm
    t_ratio = pz / max(1.0, pnl)
    if t_ratio < 0.0 or t_ratio > 1.0:
        return 0.0

    slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
    _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)

    a = max(1.5, 0.5 * W * 0.87)
    b = max(1.5, 0.5 * H * 0.87)

    u = np.clip(((px - Xc) / a + 1.0) * 0.5, 0.0, 1.0)
    v = np.clip(((py - Zc) / b + 1.0) * 0.5, 0.0, 1.0)

    D_vol, H_vol, W_vol = volume_3d.shape
    d_coord = u * (D_vol - 1.0)
    h_coord = (1.0 - v) * (H_vol - 1.0)
    w_coord = np.clip(t_ratio * (W_vol - 1.0), 0.0, W_vol - 1.0)

    coords = np.array([[d_coord], [h_coord], [w_coord]])
    val = float(ndimage.map_coordinates(volume_3d, coords, order=1, mode='nearest')[0])
    return float(np.clip(val, 0.0, 1.0))


def generate_victre_ductal_tree(kinematics, volume_3d=None, num_lobes=14, rng_seed=42):
    """
    Genera l'albero galattoforo completo secondo FDA VICTRE (duct.cxx),
    modulato e guidato in modo BIOLOGICO dalla densità 3D predetta dalla U-Net-GAN:
    - I lobi nelle regioni ad alta densità fibroglandolare (es. UOQ) sviluppano
      arborizzazioni dense con 2-3 rami secondari e grappoli di acini TDLU.
    - I lobi nelle regioni ipodense o adipose sviluppano canali più radi e sottili
      senza accumuli di TDLU (involuzione adiposa fisiologica).
    - I dotti si curvano preferenzialmente verso i gradienti di parenchima denso.
    """
    rng = np.random.RandomState(rng_seed)
    pnl = kinematics.pnl_mm
    _, _, _, Xc_apex, Zc_apex = kinematics.get_slice_parameters(kinematics.num_depth_slices - 1)

    segments = []

    for i in range(num_lobes):
        base_theta = 2.0 * np.pi * (i + rng.uniform(-0.06, 0.06)) / num_lobes

        # Test di tropismo verso parenchima più denso nell'intorno angolare
        if volume_3d is not None:
            cands_th = [base_theta - 0.14, base_theta, base_theta + 0.14]
            densities = [sample_neural_density(volume_3d, kinematics,
                         get_parenchyma_point(kinematics, 0.65, th, rho=0.55)) for th in cands_th]
            theta = cands_th[int(np.argmax(densities))]
            lobe_density = float(np.max(densities))
        else:
            theta = base_theta
            lobe_density = 0.50

        # Origine sul disco sub-areolare (seno galattoforo retroareolare)
        r_areola = rng.uniform(2.0, 4.5)
        p_origin = np.array([
            Xc_apex + r_areola * np.cos(theta),
            Zc_apex + r_areola * np.sin(theta),
            pnl - rng.uniform(1.8, 3.2)
        ])

        # Calibro modulato dalla densità ghiandolare locale
        r_trunk_start = 0.62 + 0.18 * lobe_density
        r_trunk_end = 0.46 + 0.14 * lobe_density

        # Tronco Primario: dal capezzolo alla profondità antero-media (t ~ 0.70)
        t_trunk = rng.uniform(0.66, 0.73)
        rho_trunk = rng.uniform(0.40, 0.62)
        p_trunk_end = get_parenchyma_point(kinematics, t_trunk, theta, rho=rho_trunk)

        segs0 = create_curved_tube(p_origin, p_trunk_end, r_start=r_trunk_start, r_end=r_trunk_end,
                                   num_subsegs=2, curvature=rng.uniform(-0.7, 0.7), sections=7)
        segments.extend(segs0)

        # Determinazione del grado di ramificazione in base alla densità U-Net-GAN
        # Alta densità (> 0.40): 3 rami secondari; Media: 2; Bassa (< 0.22): 1 ramo
        if lobe_density > 0.40:
            sec_branches = [-0.32, 0.0, 0.32]
        elif lobe_density > 0.22:
            sec_branches = [-0.28, 0.28]
        else:
            sec_branches = [0.0]

        for d_th_sec in sec_branches:
            theta_sec = theta + d_th_sec + rng.uniform(-0.06, 0.06)
            t_sec = rng.uniform(0.46, 0.55)
            rho_sec = np.clip(rho_trunk + np.sign(d_th_sec if d_th_sec != 0 else 1) * rng.uniform(0.06, 0.18), 0.28, 0.76)
            p_sec_end = get_parenchyma_point(kinematics, t_sec, theta_sec, rho=rho_sec)
            sec_density = sample_neural_density(volume_3d, kinematics, p_sec_end) if volume_3d is not None else 0.50

            r_sec_start = 0.40 + 0.10 * sec_density
            r_sec_end = 0.28 + 0.08 * sec_density
            segs1 = create_curved_tube(p_trunk_end, p_sec_end, r_start=r_sec_start, r_end=r_sec_end,
                                       num_subsegs=2, curvature=rng.uniform(-0.5, 0.5), sections=6)
            segments.extend(segs1)

            # Ramificazioni Terziarie e TDLU
            if sec_density > 0.35:
                tert_branches = [-0.22, 0.22]
            else:
                tert_branches = [0.0]

            for d_th_tert in tert_branches:
                theta_tert = theta_sec + d_th_tert + rng.uniform(-0.05, 0.05)
                t_tert = rng.uniform(0.24, 0.36)
                rho_tert = np.clip(rho_sec + np.sign(d_th_tert if d_th_tert != 0 else 1) * rng.uniform(0.05, 0.15), 0.22, 0.78)
                p_tert_end = get_parenchyma_point(kinematics, t_tert, theta_tert, rho=rho_tert)
                tert_density = sample_neural_density(volume_3d, kinematics, p_tert_end) if volume_3d is not None else 0.50

                r_tert_start = 0.26 + 0.08 * tert_density
                r_tert_end = 0.18 + 0.05 * tert_density
                segs2 = create_curved_tube(p_sec_end, p_tert_end, r_start=r_tert_start, r_end=r_tert_end,
                                           num_subsegs=2, curvature=rng.uniform(-0.4, 0.4), sections=5)
                segments.extend(segs2)

                # Terminal Ductal Lobular Unit (TDLU)
                # Nelle zone dense (parenchima reale), creiamo un grappolo di acini (cluster TDLU)
                if tert_density > 0.40:
                    # Grappolo di 2-3 lobuli acinari
                    num_acini = rng.randint(2, 4)
                    for k in range(num_acini):
                        offset = rng.uniform(-0.70, 0.70, size=3)
                        r_acino = rng.uniform(0.48, 0.68)
                        tdlu = trimesh.creation.icosphere(subdivisions=1, radius=r_acino)
                        tdlu.apply_translation(p_tert_end + offset)
                        segments.append(tdlu)
                elif tert_density > 0.20:
                    # Singolo lobulo TDLU standard
                    r_acino = 0.52
                    tdlu = trimesh.creation.icosphere(subdivisions=1, radius=r_acino)
                    tdlu.apply_translation(p_tert_end)
                    segments.append(tdlu)
                else:
                    # Micro-bocciolo terminale atrofico
                    tdlu = trimesh.creation.icosphere(subdivisions=1, radius=0.35)
                    tdlu.apply_translation(p_tert_end)
                    segments.append(tdlu)

    tree_mesh = trimesh.util.concatenate(segments)
    duct_color = np.array([245, 248, 252, 235], dtype=np.uint8)
    tree_mesh.visual.vertex_colors = np.tile(duct_color, (len(tree_mesh.vertices), 1))
    return tree_mesh


def generate_superficial_veins(kinematics, skin_mesh=None, rng_seed=42):
    """
    Genera il plesso venoso superficiale sottocutaneo:
    - Cerchio venoso peri-areolare di Haller (Circulus Venosus Halleri) attorno alla base areolare
    - 7 tronchi venosi superficiali con diramazioni periferiche
    Tutti i nodi sono calcolati proiettando verso l'interno lungo le normali della cute
    (profondità 3.8 mm) e vincolati rigorosamente mediante clamp_mesh_inside_skin
    per garantire matematicamente che NESSUNA vena sporga all'esterno della pelle.
    """
    rng = np.random.RandomState(rng_seed)
    pnl = kinematics.pnl_mm
    _, _, _, Xc_apex, Zc_apex = kinematics.get_slice_parameters(kinematics.num_depth_slices - 1)
    
    segments = []
    
    if skin_mesh is not None:
        def get_subcutaneous_node(target_z, target_angle, depth_inward=3.8):
            z_diff = np.abs(skin_mesh.vertices[:, 2] - target_z)
            cands = np.where(z_diff < 5.0)[0]
            if len(cands) == 0:
                cands = np.argsort(z_diff)[:60]
            
            c_x = np.mean(skin_mesh.vertices[cands, 0])
            c_y = np.mean(skin_mesh.vertices[cands, 1])
            angles = np.arctan2(skin_mesh.vertices[cands, 1] - c_y, skin_mesh.vertices[cands, 0] - c_x)
            ang_diff = np.abs(np.arctan2(np.sin(angles - target_angle), np.cos(angles - target_angle)))
            best_idx = cands[np.argmin(ang_diff)]
            
            p_skin = skin_mesh.vertices[best_idx]
            n_skin = np.array(skin_mesh.vertex_normals[best_idx], dtype=np.float64)
            norm_val = np.linalg.norm(n_skin)
            if norm_val > 1e-6:
                n_skin /= norm_val
            else:
                n_skin = np.array([0.0, 0.0, 1.0])
                
            return p_skin - depth_inward * n_skin

        # 1. Cerchio Venoso Peri-Areolare di Haller (a profondità z ~ 78% PNL, sicuro e sottocutaneo)
        ring_z = pnl * 0.78
        num_ring = 16
        ring_nodes = [get_subcutaneous_node(ring_z, -np.pi + 2.0 * np.pi * i / num_ring, depth_inward=3.8) for i in range(num_ring)]
        for i in range(num_ring):
            p1 = ring_nodes[i]
            p2 = ring_nodes[(i + 1) % num_ring]
            segs = create_curved_tube(p1, p2, 0.38, 0.38, num_subsegs=1, curvature=0.0, sections=6)
            segments.extend(segs)
            
        # 2. 7 Tronchi Venosi Superficiali nel Sottocute con passi densi e conformi
        z_steps = np.linspace(ring_z, pnl * 0.24, 7)[1:]
        num_veins = 7
        for i in range(num_veins):
            th = -np.pi + 2.0 * np.pi * i / num_veins + rng.uniform(-0.06, 0.06)
            r_idx = int(round(i * (num_ring / float(num_veins)))) % num_ring
            p_prev = ring_nodes[r_idx]
            
            r_curr = 0.44
            for z_s in z_steps:
                th_step = th + rng.uniform(-0.04, 0.04)
                p_next = get_subcutaneous_node(z_s, th_step, depth_inward=4.0)
                segs = create_curved_tube(p_prev, p_next, r_curr, r_curr * 0.92, num_subsegs=1, curvature=0.0, sections=6)
                segments.extend(segs)
                p_prev = p_next
                r_curr *= 0.92
                
            # Biforcazioni periferiche (z ~ pnl * 0.22)
            for d_th in [-0.18, 0.18]:
                th_trib = th + d_th
                p_end = get_subcutaneous_node(pnl * 0.22, th_trib, depth_inward=3.8)
                segs = create_curved_tube(p_prev, p_end, r_curr, 0.24, num_subsegs=1, curvature=0.0, sections=5)
                segments.extend(segs)
    else:
        # Fallback analitico interno (rho = 0.82 per essere sicuri al 100%)
        num_veins = 7
        for i in range(num_veins):
            th = 2.0 * np.pi * i / num_veins
            p1 = get_subcutaneous_point(kinematics, 0.78, th, rho=0.82)
            p2 = get_subcutaneous_point(kinematics, 0.50, th, rho=0.82)
            p3 = get_subcutaneous_point(kinematics, 0.20, th, rho=0.80)
            segments.extend(create_curved_tube(p1, p2, 0.42, 0.32, num_subsegs=1, curvature=0.0, sections=6))
            segments.extend(create_curved_tube(p2, p3, 0.32, 0.22, num_subsegs=1, curvature=0.0, sections=5))

    veins_mesh = trimesh.util.concatenate(segments)
    if skin_mesh is not None:
        veins_mesh = clamp_mesh_inside_skin(veins_mesh, skin_mesh, min_inward_depth=3.5)
    vein_color = np.array([30, 95, 205, 220], dtype=np.uint8)
    veins_mesh.visual.vertex_colors = np.tile(vein_color, (len(veins_mesh.vertices), 1))
    return veins_mesh


def generate_deep_arteries(kinematics, volume_3d=None, rng_seed=42):
    """
    Genera la rete arteriosa ghiandolare profonda:
    - Rami dell'Arteria Mammaria Interna (mediali)
    - Rami dell'Arteria Toracica Laterale (laterali)
    - Ramificazioni arteriolari e capillari nutrizionali guidate
      dalla densità parenchimale predetta da 3D U-Net-GAN.
    """
    rng = np.random.RandomState(rng_seed)
    pnl = kinematics.pnl_mm
    segments = []

    origins = [
        (0.12, 0.35, np.pi * 0.15),   # Tronco arterioso laterale/ascellare
        (0.12, 0.35, np.pi * 0.85),   # Tronco arterioso mediale/mammario interno
    ]

    for t_start, rho_start, theta_start in origins:
        p_root = get_parenchyma_point(kinematics, t_start, theta_start, rho=rho_start)

        # Tronco principale si dirige in avanti nel parenchima (t ~ 0.40)
        p_trunk1 = get_parenchyma_point(kinematics, 0.38, theta_start + rng.uniform(-0.15, 0.15), rho=0.50)
        segs = create_curved_tube(p_root, p_trunk1, 0.60, 0.45, num_subsegs=2, curvature=0.7, sections=6)
        segments.extend(segs)

        # Biforcazioni arteriolari verso i lobi intermedi (t ~ 0.60)
        for d_th in [-0.28, 0.28]:
            th_art = theta_start + d_th
            p_art2 = get_parenchyma_point(kinematics, 0.60, th_art, rho=0.55)
            segs2 = create_curved_tube(p_trunk1, p_art2, 0.42, 0.30, num_subsegs=2, curvature=0.5, sections=5)
            segments.extend(segs2)

            # Duttuli arteriosi terminali (t ~ 0.75)
            p_art3 = get_parenchyma_point(kinematics, 0.74, th_art + rng.uniform(-0.15, 0.15), rho=0.50)
            segs3 = create_curved_tube(p_art2, p_art3, 0.28, 0.20, num_subsegs=1, curvature=0.0, sections=5)
            segments.extend(segs3)

            # Micro-arteriole capillari per aree dense U-Net-GAN
            if volume_3d is not None:
                dens_art = sample_neural_density(volume_3d, kinematics, p_art3)
                if dens_art > 0.38:
                    for d_sub in [-0.14, 0.14]:
                        p_cap = get_parenchyma_point(kinematics, 0.80, th_art + d_sub, rho=0.48)
                        segs_cap = create_curved_tube(p_art3, p_cap, 0.20, 0.14, num_subsegs=1, curvature=0.0, sections=4)
                        segments.extend(segs_cap)

    arteries_mesh = trimesh.util.concatenate(segments)
    artery_color = np.array([215, 35, 40, 230], dtype=np.uint8)
    arteries_mesh.visual.vertex_colors = np.tile(artery_color, (len(arteries_mesh.vertices), 1))
    return arteries_mesh


def generate_nipple_areola_complex(kinematics, skin_mesh=None):
    """
    Genera il Complesso Areola-Capezzolo (NAC) con proporzioni anatomiche naturali:
    - Disco areolare conforme che aderisce perfettamente alla curvatura della cute
    - Papilla centrale del capezzolo proporzionata (r ~ 2.2 mm, h ~ 1.8 mm)
    - Tubercoli di Montgomery discreti circonferenziali
    """
    pnl = kinematics.pnl_mm
    _, _, _, Xc_apex, Zc_apex = kinematics.get_slice_parameters(kinematics.num_depth_slices - 1)
    apex_center = np.array([Xc_apex, Zc_apex, pnl])
    
    if skin_mesh is not None:
        apex_idx = int(np.argmin(np.linalg.norm(skin_mesh.vertices - apex_center, axis=1)))
        actual_apex = skin_mesh.vertices[apex_idx].copy()
        apex_normal = np.array(skin_mesh.vertex_normals[apex_idx], dtype=np.float64)
        norm_val = np.linalg.norm(apex_normal)
        if norm_val > 1e-6:
            apex_normal /= norm_val
        else:
            apex_normal = np.array([0.0, 0.0, 1.0])
            
        # 1. Disco areolare conforme ricavato dalla pelle (r ~ 12.5 mm)
        dists = np.linalg.norm(skin_mesh.vertices - actual_apex, axis=1)
        areola_v_idx = np.where(dists <= 12.5)[0]
        face_mask = np.isin(skin_mesh.faces, areola_v_idx).all(axis=1)
        areola_patch = skin_mesh.submesh([face_mask], append=True)
        # Piccolo offset microscopico (0.15 mm) per prevenire z-fighting con la cute
        areola_patch.vertices = areola_patch.vertices + areola_patch.vertex_normals * 0.15
        
        # 2. Papilla del capezzolo proporzionata
        pap_r = 2.2
        pap_h = 1.8
        cyl = trimesh.creation.cylinder(radius=pap_r, height=pap_h, sections=16)
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z_axis, apex_normal)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            rot = np.eye(4)
        else:
            ang = np.arccos(np.clip(np.dot(z_axis, apex_normal), -1.0, 1.0))
            rot = trimesh.transformations.rotation_matrix(ang, axis)
        trans = trimesh.transformations.translation_matrix(actual_apex + apex_normal * (pap_h * 0.5))
        cyl.apply_transform(trans @ rot)

        tip_dome = trimesh.creation.icosphere(subdivisions=2, radius=pap_r)
        tip_trans = trimesh.transformations.translation_matrix(actual_apex + apex_normal * pap_h)
        tip_dome.apply_transform(tip_trans)

        # 3. Tubercoli di Montgomery discreti
        tubercles = []
        for i in range(8):
            th = 2.0 * np.pi * i / 8.0
            r_tub = 8.0
            cand_pt = actual_apex + np.array([r_tub * np.cos(th), r_tub * np.sin(th), -1.0])
            t_idx = int(np.argmin(np.linalg.norm(areola_patch.vertices - cand_pt, axis=1)))
            t_pt = areola_patch.vertices[t_idx]
            t_norm = areola_patch.vertex_normals[t_idx]
            bead = trimesh.creation.icosphere(subdivisions=1, radius=0.38)
            bead.apply_translation(t_pt + t_norm * 0.12)
            tubercles.append(bead)

        nac_mesh = trimesh.util.concatenate([areola_patch, cyl, tip_dome] + tubercles)
    else:
        # Fallback procedurale proporzionato
        papilla = trimesh.creation.cylinder(radius=2.2, height=2.0, sections=16)
        papilla.apply_translation(apex_center + np.array([0.0, 0.0, 1.0]))
        areola = trimesh.creation.cylinder(radius=12.0, height=0.3, sections=24)
        areola.apply_translation(apex_center)
        nac_mesh = trimesh.util.concatenate([papilla, areola])
        
    areola_color = np.array([185, 110, 95, 240], dtype=np.uint8)
    nac_mesh.visual.vertex_colors = np.tile(areola_color, (len(nac_mesh.vertices), 1))
    return nac_mesh


def generate_mammary_nerves(kinematics, skin_mesh=None, rng_seed=42):
    """
    Genera l'innervazione sensoriale cutanea mammaria:
    - 4° Ramo Cutaneo Laterale del Nervo Intercostale (decorre attraverso il mantello adiposo verso il NAC)
    - Rami Cutanei Anteriori Mediali (3°-4° nervo intercostale peristernalmente)
    Fornisce la sensibilità fisiologica alla cute, all'areola e alla papilla.
    """
    rng = np.random.RandomState(rng_seed)
    pnl = kinematics.pnl_mm
    _, _, _, Xc_apex, Zc_apex = kinematics.get_slice_parameters(kinematics.num_depth_slices - 1)
    apex_center = np.array([Xc_apex, Zc_apex, pnl * 0.85])
    segments = []

    # 1. 4° Nervo Intercostale Laterale (principale nervo sensoriale del capezzolo)
    p_lat_origin = get_parenchyma_point(kinematics, 0.22, np.pi * 0.12, rho=0.88)
    p_lat_mid = get_parenchyma_point(kinematics, 0.55, np.pi * 0.08, rho=0.84)
    p_lat_pre = get_parenchyma_point(kinematics, 0.78, np.pi * 0.05, rho=0.72)

    segments.extend(create_curved_tube(p_lat_origin, p_lat_mid, 0.45, 0.36, num_subsegs=2, curvature=0.5, sections=5))
    segments.extend(create_curved_tube(p_lat_mid, p_lat_pre, 0.36, 0.28, num_subsegs=2, curvature=0.4, sections=5))

    # Ramificazioni terminali verso areola e capezzolo (rimangono sottocutanee)
    for d_th, offset_z in [(-0.15, -2.5), (0.0, -2.0), (0.15, -2.8)]:
        p_term = apex_center + np.array([3.5 * np.cos(d_th), 3.5 * np.sin(d_th), offset_z])
        segments.extend(create_curved_tube(p_lat_pre, p_term, 0.25, 0.15, num_subsegs=1, curvature=0.0, sections=4))

    # 2. Ramo Cutaneo Anteriore Mediale (3°-4° intercostale)
    p_med_origin = get_parenchyma_point(kinematics, 0.20, np.pi * 0.88, rho=0.88)
    p_med_mid = get_parenchyma_point(kinematics, 0.50, np.pi * 0.85, rho=0.80)
    p_med_term = get_parenchyma_point(kinematics, 0.75, np.pi * 0.80, rho=0.68)
    segments.extend(create_curved_tube(p_med_origin, p_med_mid, 0.38, 0.28, num_subsegs=2, curvature=0.4, sections=5))
    segments.extend(create_curved_tube(p_med_mid, p_med_term, 0.28, 0.18, num_subsegs=1, curvature=0.0, sections=4))

    nerve_mesh = trimesh.util.concatenate(segments)
    if skin_mesh is not None:
        nerve_mesh = clamp_mesh_inside_skin(nerve_mesh, skin_mesh, min_inward_depth=3.0)
    nerve_color = np.array([245, 225, 55, 235], dtype=np.uint8)
    nerve_mesh.visual.vertex_colors = np.tile(nerve_color, (len(nerve_mesh.vertices), 1))
    return nerve_mesh


def generate_lymphatic_vessels(kinematics, skin_mesh=None, rng_seed=42):
    """
    Genera il sistema vascolare linfatico mammario:
    - Plesso Linfatico Subareolare di Sappey (anastomosi circolare periareolare a profondità protetta)
    - Vasi Linfatici Collettori Ascellari che decorrono nel Quadrante Supero-Esterno (UOQ)
      verso i linfonodi intramammari e la stazione ascellare.
    """
    rng = np.random.RandomState(rng_seed)
    pnl = kinematics.pnl_mm
    _, _, _, Xc_apex, Zc_apex = kinematics.get_slice_parameters(kinematics.num_depth_slices - 1)
    segments = []

    # 1. Plesso di Sappey (anello linfatico subareolare, r ~ 4.8 mm, profondità sicura z ~ 80% PNL)
    r_sappey = 4.8
    num_nodes = 10
    pts_ring = []
    z_sappey = pnl * 0.80
    for i in range(num_nodes):
        th = 2.0 * np.pi * i / num_nodes
        pt = np.array([
            Xc_apex + r_sappey * np.cos(th),
            Zc_apex + r_sappey * np.sin(th),
            z_sappey + rng.uniform(-0.2, 0.2)
        ])
        pts_ring.append(pt)
    for i in range(num_nodes):
        p1 = pts_ring[i]
        p2 = pts_ring[(i + 1) % num_nodes]
        segments.extend(create_curved_tube(p1, p2, 0.24, 0.24, num_subsegs=1, curvature=0.0, sections=5))

    # 2. Vasi collettori linfatici verso il polo ascellare (UOQ: theta ~ 0.25 pi)
    p_lymph_target1 = get_parenchyma_point(kinematics, 0.42, np.pi * 0.25, rho=0.72)
    p_lymph_target2 = get_parenchyma_point(kinematics, 0.34, np.pi * 0.28, rho=0.76)

    p_start1 = pts_ring[1]
    p_mid1 = get_parenchyma_point(kinematics, 0.65, np.pi * 0.22, rho=0.80)
    segments.extend(create_curved_tube(p_start1, p_mid1, 0.32, 0.25, num_subsegs=2, curvature=0.5, sections=5))
    segments.extend(create_curved_tube(p_mid1, p_lymph_target1, 0.25, 0.20, num_subsegs=2, curvature=0.4, sections=5))

    p_start2 = pts_ring[2]
    p_mid2 = get_parenchyma_point(kinematics, 0.60, np.pi * 0.30, rho=0.82)
    segments.extend(create_curved_tube(p_start2, p_mid2, 0.30, 0.24, num_subsegs=2, curvature=0.5, sections=5))
    segments.extend(create_curved_tube(p_mid2, p_lymph_target2, 0.24, 0.18, num_subsegs=2, curvature=0.4, sections=5))

    lymph_mesh = trimesh.util.concatenate(segments)
    if skin_mesh is not None:
        lymph_mesh = clamp_mesh_inside_skin(lymph_mesh, skin_mesh, min_inward_depth=3.0)
    lymph_vessel_color = np.array([215, 235, 130, 225], dtype=np.uint8)
    lymph_mesh.visual.vertex_colors = np.tile(lymph_vessel_color, (len(lymph_mesh.vertices), 1))
    return lymph_mesh


def generate_deep_pectoral_fascia(kinematics, skin_mesh):
    """
    Genera la Fascia Pettorale Profonda (Lamina Profonda della Fascia Superficialis),
    il piano fibroso connettivale che riveste la superficie anteriore del grande pettorale
    e delimita posteriormente lo spazio di scorrimento retromammario di Chassaignac.
    """
    min_b, max_b = skin_mesh.bounds
    Nx, Ny = 50, 65
    xs = np.linspace(min_b[0], max_b[0], Nx)
    ys = np.linspace(min_b[1], max_b[1], Ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    _, W, H, Xc, Zc = kinematics.get_slice_parameters(0)
    a = max(2.0, 0.5 * W * 0.94)
    b = max(2.0, 0.5 * H * 0.94)
    p = 2.15

    dx = np.abs(X - Xc) / a
    dy = np.abs(Y - Zc) / b
    dist_se = (dx ** p + dy ** p) ** (1.0 / p)

    mask = dist_se <= 1.0
    Z = 1.2 + 1.8 * (1.0 - np.clip(dist_se, 0.0, 1.0))

    verts = []
    faces = []
    vert_map = -np.ones((Nx, Ny), dtype=int)

    for i in range(Nx):
        for j in range(Ny):
            if mask[i, j]:
                vert_map[i, j] = len(verts)
                verts.append([X[i, j], Y[i, j], Z[i, j]])

    for i in range(Nx - 1):
        for j in range(Ny - 1):
            v00 = vert_map[i, j]
            v10 = vert_map[i + 1, j]
            v01 = vert_map[i, j + 1]
            v11 = vert_map[i + 1, j + 1]
            if v00 >= 0 and v10 >= 0 and v01 >= 0 and v11 >= 0:
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

    fascia_mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
    fascia_color = np.array([230, 235, 245, 140], dtype=np.uint8)
    fascia_mesh.visual.vertex_colors = np.tile(fascia_color, (len(fascia_mesh.vertices), 1))
    return fascia_mesh


def generate_lymph_nodes(kinematics):
    """
    Genera 2 Linfonodi Intramammari reniformi (a fagiolo)
    nel Quadrante Supero-Esterno (UOQ) lungo la via di drenaggio linfatico ascellare.
    """
    pnl = kinematics.pnl_mm
    nodes = []
    
    # Posizione anatomica nel quadrante supero-esterno (UOQ: theta ~ pi/4, t ~ 0.40)
    locs = [
        (0.42, 0.72, np.pi * 0.25, 1.0),
        (0.34, 0.76, np.pi * 0.28, 0.85),
    ]
    
    for t_n, rho_n, th_n, scale_factor in locs:
        p_loc = get_parenchyma_point(kinematics, t_n, th_n, rho=rho_n)
        
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        verts = sphere.vertices.copy()
        
        # Dimensioni realistiche linfonodo: ~ 6 x 4 x 3 mm
        verts[:, 0] *= 3.0 * scale_factor
        verts[:, 1] *= 2.0 * scale_factor
        verts[:, 2] *= 1.5 * scale_factor
        
        # Ilo reniforme (indentazione)
        indent = 0.35 * np.exp(-(verts[:, 0]**2 + verts[:, 2]**2) / 3.0)
        verts[verts[:, 1] < 0, 1] += indent[verts[:, 1] < 0]
        sphere.vertices = verts
        sphere.apply_translation(p_loc)
        nodes.append(sphere)
        
    lymph_mesh = trimesh.util.concatenate(nodes)
    # Colore linfoide verde oliva chiaro / buff
    lymph_color = np.array([180, 195, 150, 225], dtype=np.uint8)
    lymph_mesh.visual.vertex_colors = np.tile(lymph_color, (len(lymph_mesh.vertices), 1))
    return lymph_mesh


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
            curved_segs = create_curved_tube(p_crest, p_skin, r_start=0.28, r_end=0.20,
                                             num_subsegs=2, curvature=0.6, sections=6)
            tubes.extend(curved_segs)
            
            # Radichette subdermiche
            v_dir = p_skin - p_crest
            v_len = np.linalg.norm(v_dir)
            if v_len > 6.0:
                p_mid = p_crest + v_dir * 0.75
                for s_sign in [-1, 1]:
                    offset = np.array([s_sign * rng.uniform(1.2, 2.5), rng.uniform(-1.0, 1.0), 0.0])
                    p_root = p_skin + offset
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
    Genera il mantello adiposo sottocutaneo / premammario volumetrico continuo (FDA VICTRE compartment.cxx).
    Riempie l'INTERO spazio anatomico tra la cute e il parenchima interno,
    eliminando qualsiasi sensazione di "guscio vuoto" ed esibendo una ricca
    trama biologica di lobuli adiposi (compartimenti di grasso).
    """
    min_b, max_b = skin_mesh.bounds
    Nx, Ny, Nz = 70, 85, 70
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

        # Mantello volumetrico solido: riempie lo spazio tra 0.83 e 0.925 (lasciando un margine dermico naturale)
        in_mantle = np.clip((dist_se - 0.83) / 0.06, 0.0, 1.0) * np.clip((0.925 - dist_se) / 0.04, 0.0, 1.0)

        # Micro-lobulazioni biologiche del tessuto adiposo (lobuli di grasso compatti)
        lobules = 0.85 + 0.15 * (
            np.sin(0.42 * X[:, :, k]) * np.cos(0.42 * Y[:, :, k]) +
            np.cos(0.32 * X[:, :, k] + 0.35 * Z[:, :, k]) * np.sin(0.35 * Y[:, :, k])
        )

        nipple_falloff = np.clip((0.95 - t_ratio) / 0.04, 0.0, 1.0) if t_ratio > 0.91 else 1.0
        base_fade = np.clip(t_ratio / 0.04, 0.0, 1.0)

        fat_grid[:, :, k] = in_mantle * lobules * nipple_falloff * base_fade

    spacing = (
        (max_b[0] - min_b[0]) / (Nx - 1),
        (max_b[1] - min_b[1]) / (Ny - 1),
        kinematics.pnl_mm / (Nz - 1)
    )
    smooth_fat = ndimage.gaussian_filter(fat_grid, sigma=0.45)
    verts, faces, normals, _ = marching_cubes(smooth_fat, level=0.35, spacing=spacing)
    verts[:, 0] += min_b[0]
    verts[:, 1] += min_b[1]

    fat_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    fat_mesh = clamp_mesh_inside_skin(fat_mesh, skin_mesh, min_inward_depth=1.5)
    fat_color = np.array([245, 230, 165, 120], dtype=np.uint8)
    fat_mesh.visual.vertex_colors = np.tile(fat_color, (len(fat_mesh.vertices), 1))
    return fat_mesh


def generate_retromammary_fat(kinematics, skin_mesh):
    """
    Genera il cuscino adiposo retromammario continuo (Spazio di Chassaignac)
    situato tra la superficie posteriore della ghiandola e la fascia del grande pettorale.
    Fornisce volume solido alla base toracica eliminando vuoti posteriori.
    """
    min_b, max_b = skin_mesh.bounds
    Nx, Ny, Nz = 60, 75, 35
    xs = np.linspace(min_b[0], max_b[0], Nx)
    ys = np.linspace(min_b[1], max_b[1], Ny)
    max_z_retro = 13.0
    zs = np.linspace(0.8, max_z_retro, Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    retro_grid = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    for k, z_val in enumerate(zs):
        t_ratio = z_val / kinematics.pnl_mm
        slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
        _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)

        a = max(1.5, 0.5 * W * 0.92)
        b = max(1.5, 0.5 * H * 0.92)
        p = 2.15

        dx = np.abs(X[:, :, k] - Xc) / a
        dy = np.abs(Y[:, :, k] - Zc) / b
        dist_se = (dx ** p + dy ** p) ** (1.0 / p)

        in_bound = np.clip((0.98 - dist_se) / 0.10, 0.0, 1.0)
        # Cuscino volumetrico compatto tra z = 1.0 e z = 11.5 mm
        z_pad = np.clip((z_val - 1.0) / 1.5, 0.0, 1.0) * np.clip((11.5 - z_val) / 2.5, 0.0, 1.0)
        lobules = 0.88 + 0.12 * np.sin(0.40 * X[:, :, k]) * np.cos(0.40 * Y[:, :, k])

        retro_grid[:, :, k] = in_bound * z_pad * lobules

    spacing = (
        (max_b[0] - min_b[0]) / (Nx - 1),
        (max_b[1] - min_b[1]) / (Ny - 1),
        (max_z_retro - 0.8) / (Nz - 1)
    )
    smooth_retro = ndimage.gaussian_filter(retro_grid, sigma=0.48)
    verts, faces, normals, _ = marching_cubes(smooth_retro, level=0.32, spacing=spacing)
    verts[:, 0] += min_b[0]
    verts[:, 1] += min_b[1]
    verts[:, 2] += 0.8

    retro_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    retro_color = np.array([245, 232, 160, 110], dtype=np.uint8)
    retro_mesh.visual.vertex_colors = np.tile(retro_color, (len(retro_mesh.vertices), 1))
    return retro_mesh


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
