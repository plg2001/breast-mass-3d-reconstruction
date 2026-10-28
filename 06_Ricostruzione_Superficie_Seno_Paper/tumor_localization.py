"""
Modulo di Localizzazione e Ricostruzione 3D della Lesione (Orientamento Verticale Anatomico Corretto)
Basato su:
- Huang et al. (2006) - 3D localization of microcalcifications/masses
- Kita, Highnam & Brady (2001) - Epipolar constraints and 3D lesion positioning
"""

import os
import cv2
import numpy as np
import trimesh
from typing import Optional, Dict, Any, Tuple
try:
    from .kinematics import BiplanarKinematics
except ImportError:
    from kinematics import BiplanarKinematics


class TumorReconstructor:
    """
    Localizza e genera la mesh 3D della lesione mammografica incrociando
    le maschere ROI della vista CC e della vista MLO con orientamento verticale anatomico.
    """

    def __init__(self, kinematics: BiplanarKinematics):
        self.kinematics = kinematics

    def localize_from_rois(
        self,
        roi_cc_path: str,
        roi_mlo_path: str
    ) -> Optional[trimesh.Trimesh]:
        """
        Estrae le ROI 2D, calcola le coordinate tridimensionali (X, Y, Z)
        e genera una mesh 3D organica della lesione collocata all'interno del seno.
        """
        if not os.path.exists(roi_cc_path) or not os.path.exists(roi_mlo_path):
            return None

        img_cc = cv2.imread(roi_cc_path, cv2.IMREAD_GRAYSCALE)
        img_mlo = cv2.imread(roi_mlo_path, cv2.IMREAD_GRAYSCALE)

        if img_cc is None or img_mlo is None:
            return None

        pts_cc = np.argwhere(img_cc > 30)
        pts_mlo = np.argwhere(img_mlo > 30)

        if len(pts_cc) == 0 or len(pts_mlo) == 0:
            return None

        y_cc_mean, x_cc_mean = pts_cc.mean(axis=0)
        y_mlo_mean, x_mlo_mean = pts_mlo.mean(axis=0)

        cc_y_span = (pts_cc[:, 0].max() - pts_cc[:, 0].min()) * self.kinematics.pixel_spacing_mm
        cc_x_span = (pts_cc[:, 1].max() - pts_cc[:, 1].min()) * self.kinematics.pixel_spacing_mm

        mlo_y_span = (pts_mlo[:, 0].max() - pts_mlo[:, 0].min()) * self.kinematics.pixel_spacing_mm
        mlo_x_span = (pts_mlo[:, 1].max() - pts_mlo[:, 1].min()) * self.kinematics.pixel_spacing_mm

        chest_cc = self.kinematics.cc_data["chest_x"]
        chest_mlo = self.kinematics.mlo_data["chest_x"]

        if self.kinematics.cc_data["chest_is_left"]:
            depth_cc_px = abs(x_cc_mean - chest_cc)
        else:
            depth_cc_px = abs(chest_cc - x_cc_mean)

        if self.kinematics.mlo_data["chest_is_left"]:
            depth_mlo_px = abs(x_mlo_mean - chest_mlo)
        else:
            depth_mlo_px = abs(chest_mlo - x_mlo_mean)

        avg_depth_px = 0.5 * (depth_cc_px + depth_mlo_px)
        tumor_depth_mm = avg_depth_px * self.kinematics.pixel_spacing_mm

        # Posizione medio-laterale (asse X, da CC)
        cc_nipple_y = float(self.kinematics.cc_data["nipple"][1])
        tumor_x = (y_cc_mean - cc_nipple_y) * self.kinematics.pixel_spacing_mm

        # Posizione supero-inferiore (asse verticale, da MLO invertito: riga < nipple_y -> +Z in alto)
        mlo_nipple_y = float(self.kinematics.mlo_data["nipple"][1])
        tumor_h = (mlo_nipple_y - y_mlo_mean) * self.kinematics.pixel_spacing_mm

        # Semi-assi del tumore
        rx = max(2.5, 0.5 * cc_y_span)
        rh = max(2.5, 0.5 * mlo_y_span)
        rdepth = max(2.5, 0.25 * (cc_x_span + mlo_x_span))

        base_sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        verts = base_sphere.vertices.copy()

        # Coordinate upright: [X_width, Y_height, Z_depth]
        verts[:, 0] *= rx
        verts[:, 1] *= rh
        verts[:, 2] *= rdepth

        # Piccola ondulazione biologica
        seed_offsets = 1.0 + 0.08 * np.sin(3.0 * verts[:, 0]) * np.cos(3.0 * verts[:, 1])
        verts = verts * seed_offsets[:, np.newaxis]

        # Traslazione nella posizione 3D calcolata all'interno del seno eretto
        verts[:, 0] += tumor_x
        verts[:, 1] += tumor_h
        verts[:, 2] += tumor_depth_mm

        tumor_mesh = trimesh.Trimesh(vertices=verts, faces=base_sphere.faces, process=True)
        tumor_mesh.visual.vertex_colors = [220, 30, 45, 255]

        return tumor_mesh
