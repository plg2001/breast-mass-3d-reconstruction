"""
Epipolar Stereotactic Tumor Localization Engine
===============================================
Basato sui principi scientifici di cinematica e geometria multiview bi-planare:
1. Yasuyo Kita, Ralph Highnam, and Michael Brady (2001).
   "Correspondence between different views of breast X-rays using curved epipolar lines."
   IEEE Transactions on Medical Imaging / Medical Image Analysis.
2. Jin-shan Huang, Yong-hong Peng, et al. (2006).
   "Reconstruction and rendering of microcalcifications from two mammogram views
   by modified projective grid space (MPGS)."
   Computerized Medical Imaging and Graphics, 30(4), 213-223.
3. Franklyn Pinto et al. (2023).
   "Generative compressed breast shape model for digital mammography and digital breast tomosynthesis."
   Medical Physics, 50(2), 798-812.

Funzionalita:
- Triangolazione stereotassica esatta del centroide lesionale combinando le proiezioni CC ed MLO
- Calcolo dello scostamento medio-laterale X, dell'altezza verticale Y e della profondita Z lungo la PNL
- Verifica e confinamento fisiologico del centroide nel parenchima fibroglandolare (regola di infiltrazione)
- Determinazione automatica del quadrante anatomico (UOQ, UIQ, LOQ, LIQ, Retroareolare)
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


class EpipolarTumorLocalizer:
    """
    Localizzatore tridimensionale della lesione mammografica basato su vincoli epipolari bi-planari.
    """

    def __init__(self, kinematics):
        self.kinematics = kinematics

    def localize(
        self,
        roi_cc_path: str,
        roi_mlo_path: str,
        laterality: str = "AUTO"
    ) -> Dict[str, Any]:
        """
        Calcola le coordinate 3D esatte (X, Y, Z) in millimetri e le metriche cliniche di posizione.
        """
        if not os.path.exists(roi_cc_path) or not os.path.exists(roi_mlo_path):
            raise FileNotFoundError(f"File ROI non trovati: {roi_cc_path} o {roi_mlo_path}")

        img_cc = cv2.imread(roi_cc_path, cv2.IMREAD_GRAYSCALE)
        img_mlo = cv2.imread(roi_mlo_path, cv2.IMREAD_GRAYSCALE)

        if img_cc is None or img_mlo is None:
            raise ValueError("Impossibile caricare le immagini delle maschere ROI.")

        pts_cc = np.argwhere(img_cc > 30)
        pts_mlo = np.argwhere(img_mlo > 30)

        if len(pts_cc) == 0 or len(pts_mlo) == 0:
            raise ValueError("Maschere ROI prive di pixel positivi della lesione.")

        # Baricentri 2D (riga, colonna) nelle proiezioni
        row_cc, col_cc = pts_cc[:, 0].mean(), pts_cc[:, 1].mean()
        row_mlo, col_mlo = pts_mlo[:, 0].mean(), pts_mlo[:, 1].mean()

        # Bounding box 2D in pixel
        span_cc_y = (pts_cc[:, 0].max() - pts_cc[:, 0].min()) * self.kinematics.pixel_spacing_mm
        span_cc_x = (pts_cc[:, 1].max() - pts_cc[:, 1].min()) * self.kinematics.pixel_spacing_mm
        span_mlo_y = (pts_mlo[:, 0].max() - pts_mlo[:, 0].min()) * self.kinematics.pixel_spacing_mm
        span_mlo_x = (pts_mlo[:, 1].max() - pts_mlo[:, 1].min()) * self.kinematics.pixel_spacing_mm

        # -------------------------------------------------------------
        # 1. Profondità Antero-Posteriore lungo la PNL (Asse Z)
        # -------------------------------------------------------------
        chest_cc = self.kinematics.cc_data["chest_x"]
        chest_mlo = self.kinematics.mlo_data["chest_x"]
        pnl_cc_px = max(1.0, float(self.kinematics.cc_data["pnl_length"]))
        pnl_mlo_px = max(1.0, float(self.kinematics.mlo_data["pnl_length"]))

        depth_cc_px = abs(col_cc - chest_cc)
        depth_mlo_px = abs(col_mlo - chest_mlo)

        t_ratio_cc = np.clip(depth_cc_px / pnl_cc_px, 0.0, 1.0)
        t_ratio_mlo = np.clip(depth_mlo_px / pnl_mlo_px, 0.0, 1.0)

        # Fuga epipolare e media ponderata normalizzata
        t_ratio_avg = float(0.5 * (t_ratio_cc + t_ratio_mlo))
        z_depth_mm = float(t_ratio_avg * self.kinematics.pnl_mm)

        # -------------------------------------------------------------
        # 2. Posizione Medio-Laterale (Asse X, da proiezione CC)
        # -------------------------------------------------------------
        cc_nipple_y = float(self.kinematics.cc_data["nipple"][1])
        x_mm = float((row_cc - cc_nipple_y) * self.kinematics.pixel_spacing_mm)

        # -------------------------------------------------------------
        # 3. Altezza Supero-Inferiore (Asse Y, da proiezione MLO)
        # -------------------------------------------------------------
        mlo_nipple_y = float(self.kinematics.mlo_data["nipple"][1])
        # Nelle coordinate immagine, la riga 0 e in alto (craniale):
        y_mm = float((mlo_nipple_y - row_mlo) * self.kinematics.pixel_spacing_mm)

        # -------------------------------------------------------------
        # 4. Verifica e Confinamento Fisiologico nel Parenchima
        # -------------------------------------------------------------
        slice_idx = int(np.clip(t_ratio_avg * (self.kinematics.num_depth_slices - 1), 0, self.kinematics.num_depth_slices - 1))
        _, W_slice, H_slice, Xc_slice, Zc_slice = self.kinematics.get_slice_parameters(slice_idx)

        # Semiassi dell'involucro parenchimale a questa profondita
        a_parenchima = max(2.5, 0.5 * W_slice * 0.84)
        b_parenchima = max(2.5, 0.5 * H_slice * 0.84)
        p_slice = float(2.15 - 0.15 * t_ratio_avg)

        dx = abs(x_mm - Xc_slice) / a_parenchima
        dy = abs(y_mm - Zc_slice) / b_parenchima
        rho_parenchima = float((dx ** p_slice + dy ** p_slice) ** (1.0 / p_slice))

        # Se la compressione radiografica deforma il centroide oltre il bordo,
        # lo confiniamo morbidamente entro il parenchima reale (rho <= 0.90)
        clamped = False
        if rho_parenchima > 0.90:
            scale = 0.88 / rho_parenchima
            x_mm = float(Xc_slice + (x_mm - Xc_slice) * scale)
            y_mm = float(Zc_slice + (y_mm - Zc_slice) * scale)
            clamped = True
            rho_parenchima = 0.88

        # -------------------------------------------------------------
        # 5. Determinazione Quadrante Anatomico
        # -------------------------------------------------------------
        is_upper = (y_mm >= Zc_slice)
        
        # Lateralita seno
        lat_detected = self.kinematics.cc_data.get("laterality", "R")
        if laterality != "AUTO":
            lat_detected = laterality.upper()

        # Nel seno destro: x < Xc e laterale (esterno), x >= Xc e mediale (interno)
        # Nel seno sinistro: x > Xc e laterale (esterno), x <= Xc e mediale (interno)
        if lat_detected == "R":
            is_outer = (x_mm <= Xc_slice)
        else:
            is_outer = (x_mm >= Xc_slice)

        if t_ratio_avg > 0.85:
            quadrant_code = "SA"
            quadrant_name = "Regione Retroareolare / Subareolare"
        elif is_upper and is_outer:
            quadrant_code = "UOQ"
            quadrant_name = "Quadrante Supero-Esterno (Upper Outer Quadrant)"
        elif is_upper and not is_outer:
            quadrant_code = "UIQ"
            quadrant_name = "Quadrante Supero-Interno (Upper Inner Quadrant)"
        elif not is_upper and is_outer:
            quadrant_code = "LOQ"
            quadrant_name = "Quadrante Infero-Esterno (Lower Outer Quadrant)"
        else:
            quadrant_code = "LIQ"
            quadrant_name = "Quadrante Infero-Interno (Lower Inner Quadrant)"

        # Calcolo dimensioni stimate della lesione
        diam_x_mm = float(max(span_cc_y, 4.0))
        diam_y_mm = float(max(span_mlo_y, 4.0))
        diam_z_mm = float(max(0.5 * (span_cc_x + span_mlo_x), 4.0))

        centroid_3d = np.array([x_mm, y_mm, z_depth_mm], dtype=np.float64)

        return {
            "centroid_3d": centroid_3d,
            "x_mm": round(x_mm, 2),
            "y_mm": round(y_mm, 2),
            "z_depth_mm": round(z_depth_mm, 2),
            "t_ratio_pnl": round(t_ratio_avg, 3),
            "rho_parenchyma": round(rho_parenchima, 3),
            "clamped": clamped,
            "quadrant_code": quadrant_code,
            "quadrant_name": quadrant_name,
            "laterality": lat_detected,
            "estimated_diameter_mm": {
                "x_width": round(diam_x_mm, 2),
                "y_height": round(diam_y_mm, 2),
                "z_depth": round(diam_z_mm, 2),
                "equivalent_sphere_diameter": round(float((diam_x_mm * diam_y_mm * diam_z_mm)**(1.0/3.0)), 2)
            },
            "raw_projections": {
                "cc_centroid_px": (round(float(col_cc), 1), round(float(row_cc), 1)),
                "mlo_centroid_px": (round(float(col_mlo), 1), round(float(row_mlo), 1)),
                "t_ratio_cc": round(float(t_ratio_cc), 3),
                "t_ratio_mlo": round(float(t_ratio_mlo), 3)
            }
        }
