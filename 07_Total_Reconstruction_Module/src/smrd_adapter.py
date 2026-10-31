"""
SMRD Adapter Module (De Sisternes et al. 2015)
==============================================
Interfaccia verso il motore SMRD (Spherical-Harmonic Multi-view Radial Deformation
con Modello di Spicolatura 3D) dal Modulo 04.

Produce la mesh chiusa, spicolata e manifold del tumore mammario.
"""

import os
import sys
import cv2
import trimesh
import numpy as np
from typing import Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODULE_04_DIR = os.path.join(PROJECT_ROOT, "04_Spicular_Lesion_SMRD_Module")

if MODULE_04_DIR not in sys.path:
    sys.path.insert(0, MODULE_04_DIR)

try:
    from smrd_lesion_reconstruction import SMRDReconstructor
except ImportError as e:
    raise ImportError(f"Impossibile importare SMRDReconstructor da {MODULE_04_DIR}: {e}")


def reconstruct_smrd_lesion(
    roi_cc_path: str,
    roi_mlo_path: str,
    patient_id: str = "P_00000",
    spicularity: float = 1.25,
    num_spikes: int = 48,
    pixel_spacing: float = 0.05,
    grid_resolution: int = 120
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Esegue la ricostruzione 3D della sola lesione tumorale con il metodo SMRD.
    Ritorna la mesh centrata e il dizionario delle metriche cliniche e biometriche.
    """
    if not os.path.exists(roi_cc_path) or not os.path.exists(roi_mlo_path):
        raise FileNotFoundError(f"File ROI non trovati: {roi_cc_path} o {roi_mlo_path}")

    img_cc = cv2.imread(roi_cc_path, cv2.IMREAD_GRAYSCALE)
    img_mlo = cv2.imread(roi_mlo_path, cv2.IMREAD_GRAYSCALE)

    if img_cc is None or img_mlo is None:
        raise ValueError("Impossibile caricare le maschere ROI.")

    reconstructor = SMRDReconstructor(
        l_max=6,
        beta_deg=45.0,
        smooth_lam=0.08,
        blend_weight=0.72,
        pixel_spacing=pixel_spacing,
        grid_theta=grid_resolution,
        grid_phi=grid_resolution * 2,
        spicularity=spicularity,
        num_spikes_3d=num_spikes,
        spike_sharpness=0.038
    )

    mesh, metrics = reconstructor.reconstruct(img_cc, img_mlo, patient_id=patient_id)
    
    # Assegna colore tumorale malizioso scuro con componente ematica
    tumor_color = np.array([165, 25, 45, 255], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(tumor_color, (len(mesh.vertices), 1))
    
    return mesh, metrics
