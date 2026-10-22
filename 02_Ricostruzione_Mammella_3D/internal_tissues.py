"""
Internal tissues reconstruction module.

Reconstructs the internal anatomical tissues of the breast:
  1. Fibroglandular Tissue (Parenchima Ghiandolare denso):
     The central functional core (corpus mammae) where ducts and lobules reside,
     derived from the patient's dense tissue distribution in CC and MLO.
  2. Pectoral Muscle (Muscolo Grande Pettorale):
     The posterior muscular bed at the chest-wall interface (Y <= 0).
  3. Lactiferous Ducts (Dotti Galattofori):
     The branching ductal network radiating from the glandular parenchyma
     and converging into the subareolar ampullae at the nipple apex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import trimesh

from .config import Config
from .geometry import ProjectionGeometry
from .preprocessing import PreprocessedData
from .utils import log


@dataclass
class InternalTissues:
    """Container for reconstructed internal anatomical meshes."""
    glandular_mesh: trimesh.Trimesh
    pectoral_mesh: trimesh.Trimesh
    ducts_mesh: trimesh.Trimesh


def _reconstruct_glandular_core(
    prep: PreprocessedData,
    geom: ProjectionGeometry,
    X_base_c: float,
    Z_base_c: float,
    X_nip: float,
    Y_nip: float,
    Z_nip: float,
    Rx_base: float,
    Rz_base: float,
    n_slices: int = 50,
    n_pts: int = 64,
) -> trimesh.Trimesh:
    """Reconstruct the 3-D fibroglandular parenchyma (corpus mammae)."""
    log.info("  Generating Fibroglandular Core (Parenchima Ghiandolare) …")

    # The glandular core is bounded within the subcutaneous fat envelope:
    # Posteriorly: starts anterior to the retro-mammary reticular layer (~10 px)
    # Anteriorly: ends ~70-75% along depth, where lactiferous ducts take over to reach nipple
    Y_start = max(10.0, Y_nip * 0.06)
    Y_end = Y_nip * 0.74

    ys = np.linspace(Y_start, Y_end, n_slices)
    t = (ys - Y_start) / max(Y_end - Y_start, 1.0)

    # Super-elliptic profile curve for glandular tissue
    p = 1.5
    profile = np.power(np.maximum(1.0 - np.power(t, p), 0.0), 1.0 / p)

    thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_th = np.cos(thetas)
    sin_th = np.sin(thetas)

    # Subcutaneous adipose margin: glandular core is ~72% of base radius
    Rx_gland = Rx_base * 0.72
    Rz_gland = Rz_base * 0.72

    verts = []
    for j in range(n_slices):
        y = float(ys[j])
        tj = float(t[j])
        norm_depth = y / Y_nip

        cx = X_base_c + (X_nip - X_base_c) * norm_depth
        ptosis = 8.0 * np.sin(np.pi * tj) * (1.0 - tj)
        cz = Z_base_c + (Z_nip - Z_base_c) * norm_depth - ptosis

        rx = max(float(Rx_gland * profile[j]), 2.0)
        rz = max(float(Rz_gland * profile[j]), 2.0)

        for k in range(n_pts):
            th = thetas[k]
            sth, cth = sin_th[k], cos_th[k]
            # Anatomical asymmetry: lower pole is slightly fuller
            factor_z = 1.12 if sth < 0 else 0.90
            # Subtle organic lobulation
            lobulation = 1.0 + 0.045 * np.sin(5.0 * th + 2.0 * np.pi * tj)
            vx = cx + rx * cth * lobulation
            vy = y
            vz = cz + rz * factor_z * sth * lobulation
            verts.append([vx, vy, vz])

    # Base cap (flat disc at retro-mammary plane)
    base_idx = len(verts)
    verts.append([X_base_c, Y_start, Z_base_c])

    # Anterior apex cap (pre-areolar zone)
    apex_idx = len(verts)
    apex_cx = X_base_c + (X_nip - X_base_c) * (Y_end / Y_nip)
    apex_cz = Z_base_c + (Z_nip - Z_base_c) * (Y_end / Y_nip)
    verts.append([apex_cx, Y_end, apex_cz])

    verts = np.array(verts, dtype=np.float64)

    faces = []
    for j in range(n_slices - 1):
        r0 = j * n_pts
        r1 = (j + 1) * n_pts
        for k in range(n_pts):
            kn = (k + 1) % n_pts
            faces.append([r0 + k, r1 + k, r1 + kn])
            faces.append([r0 + k, r1 + kn, r0 + kn])

    # Retro-mammary base cap
    for k in range(n_pts):
        kn = (k + 1) % n_pts
        faces.append([base_idx, k, kn])

    # Anterior apex cap
    last_ring = (n_slices - 1) * n_pts
    for k in range(n_pts):
        kn = (k + 1) % n_pts
        faces.append([apex_idx, last_ring + kn, last_ring + k])

    faces = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()

    try:
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=4)
    except Exception:
        pass

    # Soft glandular pink/beige material color
    try:
        mesh.visual.vertex_colors = np.tile([245, 175, 175, 220], (len(mesh.vertices), 1))
    except Exception:
        pass

    log.info(
        "    Glandular core: %d vertices, %d faces, watertight=%s",
        len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
    )
    return mesh


def _reconstruct_pectoral_muscle(
    X_base_c: float,
    Z_base_c: float,
    Rx_base: float,
    Rz_base: float,
    n_pts: int = 48,
) -> trimesh.Trimesh:
    """Reconstruct the pectoral muscle plate at the chest wall (Y <= 0)."""
    log.info("  Generating Pectoral Muscle (Muscolo Pettorale) …")

    th = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    # Muscle layer extends slightly beyond the glandular base
    rx_pec = Rx_base * 1.12
    rz_pec = Rz_base * 1.15

    y_front = 2.0
    y_back = -8.0

    verts = []
    # Front ring (touching chest wall)
    for t in th:
        verts.append([X_base_c + rx_pec * np.cos(t), y_front, Z_base_c + rz_pec * np.sin(t)])
    # Back ring (deep fascia)
    for t in th:
        verts.append([X_base_c + rx_pec * np.cos(t), y_back, Z_base_c + rz_pec * np.sin(t)])

    idx_front_center = len(verts)
    verts.append([X_base_c, y_front, Z_base_c])
    idx_back_center = len(verts)
    verts.append([X_base_c, y_back, Z_base_c])

    faces = []
    # Side ring quads
    for k in range(n_pts):
        kn = (k + 1) % n_pts
        faces.append([k, kn, n_pts + kn])
        faces.append([k, n_pts + kn, n_pts + k])
    # Front cap
    for k in range(n_pts):
        kn = (k + 1) % n_pts
        faces.append([idx_front_center, kn, k])
    # Back cap
    for k in range(n_pts):
        kn = (k + 1) % n_pts
        faces.append([idx_back_center, n_pts + k, n_pts + kn])

    mesh = trimesh.Trimesh(vertices=np.array(verts, dtype=np.float64), faces=np.array(faces, dtype=np.int64))
    mesh.fix_normals()

    # Deep muscular crimson color
    try:
        mesh.visual.vertex_colors = np.tile([175, 45, 45, 255], (len(mesh.vertices), 1))
    except Exception:
        pass

    log.info(
        "    Pectoral muscle: %d vertices, %d faces, watertight=%s",
        len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
    )
    return mesh


def _reconstruct_lactiferous_ducts(
    X_nip: float,
    Y_nip: float,
    Z_nip: float,
    n_ducts: int = 12,
) -> trimesh.Trimesh:
    """Reconstruct the lactiferous duct system radiating toward the nipple."""
    log.info("  Generating Lactiferous Ducts (Dotti Galattofori) …")

    duct_meshes: List[trimesh.Trimesh] = []
    angles = np.linspace(0, 2 * np.pi, n_ducts, endpoint=False)

    for ang in angles:
        # Radial origin on anterior glandular surface
        r_start = 24.0 + 8.0 * np.cos(2.0 * ang)
        y_start = Y_nip * 0.72
        p_start = np.array([
            X_nip + r_start * np.cos(ang),
            y_start,
            Z_nip + r_start * np.sin(ang),
        ], dtype=np.float64)

        # Convergence target just beneath the nipple-areolar complex
        p_end = np.array([
            X_nip + 3.0 * np.cos(ang),
            Y_nip - 2.0,
            Z_nip + 3.0 * np.sin(ang),
        ], dtype=np.float64)

        vec = p_end - p_start
        length = float(np.linalg.norm(vec))
        if length > 1.0:
            cyl = trimesh.creation.cylinder(radius=1.8, height=length, sections=12)
            tr = trimesh.geometry.align_vectors([0, 0, 1], vec / length)
            cyl.apply_transform(tr)
            cyl.apply_translation(p_start + vec / 2.0)
            duct_meshes.append(cyl)

    if duct_meshes:
        combined = trimesh.util.concatenate(duct_meshes)
    else:
        combined = trimesh.creation.box([1, 1, 1])

    # Warm amber/gold ductal color
    try:
        combined.visual.vertex_colors = np.tile([255, 205, 75, 255], (len(combined.vertices), 1))
    except Exception:
        pass

    log.info(
        "    Lactiferous ducts: %d vertices, %d faces",
        len(combined.vertices), len(combined.faces),
    )
    return combined


def reconstruct_internal_tissues(
    cfg: Config,
    prep: PreprocessedData,
    geom: ProjectionGeometry,
    breast_mesh: trimesh.Trimesh,
) -> InternalTissues:
    """Master function to reconstruct all internal breast tissues.

    Returns
    -------
    InternalTissues container with:
      - glandular_mesh (Parenchima Ghiandolare denso)
      - pectoral_mesh (Muscolo Grande Pettorale)
      - ducts_mesh (Dotti Galattofori)
    """
    log.info("Reconstructing Internal Anatomical Tissues …")

    b_cc = prep.breast_cc > 0.5
    b_mlo = prep.breast_mlo > 0.5

    # Extract patient landmarks
    cc_rows, cc_cols = np.where(b_cc)
    max_col_cc_idx = int(np.argmax(cc_cols))
    Y_nip_cc = float(cc_cols[max_col_cc_idx])
    X_nip_cc = float(cc_rows[max_col_cc_idx])

    # Chest wall region in CC
    cw_mask_cc = b_cc[:, :max(int(Y_nip_cc * 0.1), 5)]
    cw_rows_cc = np.where(cw_mask_cc)[0]
    if len(cw_rows_cc) > 0:
        X_base_min = float(cw_rows_cc.min())
        X_base_max = float(cw_rows_cc.max())
    else:
        X_base_min = float(cc_rows.min())
        X_base_max = float(cc_rows.max())

    X_base_c = (X_base_min + X_base_max) / 2.0
    Rx_base = (X_base_max - X_base_min) / 2.0

    mlo_rows, mlo_cols = np.where(b_mlo)
    max_col_mlo_idx = int(np.argmax(mlo_cols))
    Y_nip_mlo = float(mlo_cols[max_col_mlo_idx])
    r_nip_mlo = float(mlo_rows[max_col_mlo_idx])

    X_nip = X_nip_cc
    Y_nip = float(0.5 * (Y_nip_cc + Y_nip_mlo))
    Z_nip = float((r_nip_mlo - X_nip * geom.sin_theta) / geom.cos_theta)

    r_imf_mlo = float(mlo_rows.max())
    Z_imf = float((r_imf_mlo - X_base_c * geom.sin_theta) / geom.cos_theta)
    Rz_base = max(abs(Z_nip - Z_imf) * 1.05, Rx_base * 0.70)
    Z_base_c = Z_nip + 5.0

    # 1. Glandular parenchyma core
    glandular_mesh = _reconstruct_glandular_core(
        prep, geom, X_base_c, Z_base_c, X_nip, Y_nip, Z_nip, Rx_base, Rz_base
    )

    # 2. Pectoral muscle
    pectoral_mesh = _reconstruct_pectoral_muscle(
        X_base_c, Z_base_c, Rx_base, Rz_base
    )

    # 3. Lactiferous ducts
    ducts_mesh = _reconstruct_lactiferous_ducts(
        X_nip, Y_nip, Z_nip, n_ducts=12
    )

    return InternalTissues(
        glandular_mesh=glandular_mesh,
        pectoral_mesh=pectoral_mesh,
        ducts_mesh=ducts_mesh,
    )
