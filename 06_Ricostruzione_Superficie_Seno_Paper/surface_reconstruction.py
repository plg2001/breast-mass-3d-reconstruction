"""
Modulo di Ricostruzione della Superficie 3D della Mammella (Orientamento Verticale Anatomico)
Basato sui modelli di:
- Pinto et al. (2023) - Generative compressed breast shape model (Super-ellissi e profili p(t))
- Rodríguez-Ruiz et al. (2017) - In vivo shape characterization (Curvatura cutanea radiale)
- Huang et al. (2006) - Raccordo anatomico alla cassa toracica e mesh poligonale continua
"""

import numpy as np
import trimesh
from typing import Dict, Any, Tuple
try:
    from .kinematics import BiplanarKinematics
except ImportError:
    from kinematics import BiplanarKinematics


class BreastSurfaceReconstructor:
    """
    Costruisce una mesh tridimensionale continua, manifold e chiusa (watertight)
    della superficie cutanea esterna della mammella con orientamento VERTICALE ANATOMICO
    (seno eretto: altezza supero-inferiore lungo l'asse verticale Z di Blender,
    larghezza medio-laterale lungo X, proiezione capezzolo in avanti lungo Y).
    """

    def __init__(
        self,
        kinematics: BiplanarKinematics,
        num_angular_samples: int = 64,
        superellipse_exponent_base: float = 2.15
    ):
        self.kinematics = kinematics
        self.num_angular_samples = num_angular_samples
        self.p_base = superellipse_exponent_base

    def reconstruct_surface_mesh(self) -> trimesh.Trimesh:
        """
        Esegue il lofting super-ellittico continuo e genera la mesh 3D completa.
        Coordinate glTF/Blender:
        - Asse 0 (X): Larghezza medio-laterale (CC)
        - Asse 1 (Y in glTF -> Z verticale in Blender): Altezza supero-inferiore (MLO)
        - Asse 2 (Z in glTF -> Y profondità in Blender): Profondità torace-capezzolo (PNL)
        """
        Ny = self.kinematics.num_depth_slices
        Ntheta = self.num_angular_samples
        theta = np.linspace(0.0, 2.0 * np.pi, Ntheta, endpoint=False)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        sgn_cos = np.sign(cos_t)
        sgn_sin = np.sign(sin_t)
        abs_cos = np.abs(cos_t)
        abs_sin = np.abs(sin_t)

        all_verts = []

        for i in range(Ny):
            y_depth, W, H, Xc, Zc = self.kinematics.get_slice_parameters(i)
            t_ratio = self.kinematics.t[i]

            # Esponente super-ellittico p(t) secondo Pinto et al. (2023)
            p = self.p_base - 0.15 * t_ratio
            exp_val = 2.0 / p

            a = 0.5 * W  # semiasse orizzontale (larghezza)
            b = 0.5 * H  # semiasse verticale (altezza)

            # Coordinate perimetrali della sezione
            x_ring = Xc + a * sgn_cos * np.power(abs_cos, exp_val)
            h_ring = Zc + b * sgn_sin * np.power(abs_sin, exp_val)  # altezza verticale
            depth_ring = np.full_like(x_ring, y_depth)              # proiezione in avanti

            # Mapping glTF: [X_width, Y_height, Z_depth]
            ring = np.stack([x_ring, h_ring, depth_ring], axis=1)
            all_verts.extend(ring)

        all_faces = []

        # Connessione tubolare ad anelli (quad grid -> 2 triangoli)
        for i in range(Ny - 1):
            for j in range(Ntheta):
                jn = (j + 1) % Ntheta
                v0 = i * Ntheta + j
                v1 = i * Ntheta + jn
                v2 = (i + 1) * Ntheta + j
                v3 = (i + 1) * Ntheta + jn

                all_faces.append([v0, v1, v2])
                all_faces.append([v1, v3, v2])

        # Chiusura posteriore sulla parete toracica (depth=0)
        base_center_x = float(self.kinematics.Xc_mm[0])
        base_center_h = float(self.kinematics.Zc_mm[0])
        base_center_depth = 0.0
        base_idx = len(all_verts)
        all_verts.append([base_center_x, base_center_h, base_center_depth])

        for j in range(Ntheta):
            jn = (j + 1) % Ntheta
            all_faces.append([base_idx, jn, j])

        # Chiusura anteriore sull'apice a cupola del capezzolo (depth=max)
        apex_center_x = float(self.kinematics.Xc_mm[-1])
        apex_center_h = float(self.kinematics.Zc_mm[-1])
        apex_center_depth = float(self.kinematics.depth_y_mm[-1] + 3.0)
        apex_idx = len(all_verts)
        all_verts.append([apex_center_x, apex_center_h, apex_center_depth])

        last_ring_start = (Ny - 1) * Ntheta
        for j in range(Ntheta):
            jn = (j + 1) % Ntheta
            all_faces.append([apex_idx, last_ring_start + j, last_ring_start + jn])

        # Creazione della mesh con trimesh
        mesh = trimesh.Trimesh(
            vertices=np.array(all_verts, dtype=np.float64),
            faces=np.array(all_faces, dtype=np.int64),
            process=True
        )

        # Filtro Laplaciano conservativo per eliminare pieghe e transizioni brusche
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.35, iterations=2)

        # Garantisci volume positivo e normali consistenti rivolte all'esterno
        if mesh.volume < 0:
            mesh.faces = np.fliplr(mesh.faces)
        trimesh.repair.fix_normals(mesh)

        return mesh

    def compute_anatomical_metrics(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Calcola i parametri metrici e volumetrici della mesh ricostruita.
        """
        bbox_min, bbox_max = mesh.bounds
        dimensions = bbox_max - bbox_min

        volume_cm3 = abs(float(mesh.volume)) / 1000.0 if mesh.is_watertight else float(mesh.convex_hull.volume) / 1000.0
        area_cm2 = float(mesh.area) / 100.0

        return {
            "volume_cm3": round(volume_cm3, 2),
            "surface_area_cm2": round(area_cm2, 2),
            "width_mm": round(float(dimensions[0]), 2),       # X: larghezza
            "height_vertical_mm": round(float(dimensions[1]), 2), # Y (glTF) / Z (Blender): altezza
            "depth_pnl_mm": round(float(dimensions[2]), 2),   # Z (glTF) / Y (Blender): proiezione
            "is_watertight": bool(mesh.is_watertight),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
        }
