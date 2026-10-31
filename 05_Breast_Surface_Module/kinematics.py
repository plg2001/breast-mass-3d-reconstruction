"""
Modulo di Allineamento Cinematico Bi-Planare e Normalizzazione PNL
Basato su:
- Huang et al. (2006) - Allineamento epipolare e spazio MPGS
- Kita, Highnam & Brady (2001) - Vincoli di corrispondenza multi-view
- Pinto et al. (2023) - Spessore e parametri dimensionali metrologici
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Any, Tuple


class BiplanarKinematics:
    """
    Gestisce l'allineamento spaziale e la calibrazione metrica tra le viste
    ortogonali CC (Cranio-Caudale) e MLO (Medio-Laterale Obliqua).
    Garantisce l'orientamento verticale anatomico corretto:
    - Polo superiore (craniale, pettorale) rivolto verso l'alto (+Z)
    - Polo inferiore (caudale, solco sottomammario) rivolto verso il basso (-Z)
    """

    def __init__(
        self,
        cc_data: Dict[str, Any],
        mlo_data: Dict[str, Any],
        pixel_spacing_mm: float = 0.05,
        num_depth_slices: int = 64
    ):
        self.cc_data = cc_data
        self.mlo_data = mlo_data
        self.pixel_spacing_mm = pixel_spacing_mm
        self.num_depth_slices = num_depth_slices

        # Determina la lunghezza PNL condivisa (in pixel e in mm)
        pnl_cc = cc_data["pnl_length"]
        pnl_mlo = mlo_data["pnl_length"]
        self.pnl_px = max(pnl_cc, pnl_mlo)
        self.pnl_mm = self.pnl_px * self.pixel_spacing_mm

        # Asse di campionamento t in [0.0, 0.985]
        self.t = np.linspace(0.0, 0.985, self.num_depth_slices)
        self.depth_y_mm = self.t * self.pnl_mm

        self._interpolate_and_smooth_profiles()

    def _interpolate_and_smooth_profiles(self):
        """
        Interpolazione e filtraggio dei profili di sezione trasversale W(t) e H(t).
        Inverte l'asse delle righe dell'immagine MLO per far corrispondere il polo craniale
        all'asse verticale positivo (+Z).
        """
        cc_ratios = self.cc_data["depth_ratios"]
        cc_widths = self.cc_data["widths"]
        cc_centers = self.cc_data["centers"]

        mlo_ratios = self.mlo_data["depth_ratios"]
        mlo_widths = self.mlo_data["widths"]
        mlo_centers = self.mlo_data["centers"]

        # Interpola lungo l'asse comune t
        w_interp = np.interp(self.t, cc_ratios, cc_widths)
        xc_interp = np.interp(self.t, cc_ratios, cc_centers)

        h_interp = np.interp(self.t, mlo_ratios, mlo_widths)
        zc_interp = np.interp(self.t, mlo_ratios, mlo_centers)

        cc_nipple_y = float(self.cc_data["nipple"][1])
        mlo_nipple_y = float(self.mlo_data["nipple"][1])

        # Asse orizzontale (medio-laterale) da CC
        xc_rel = xc_interp - cc_nipple_y

        # Asse verticale (supero-inferiore) da MLO:
        # Nelle immagini digitali, la riga 0 è in alto (craniale), quindi (nipple_y - riga)
        # è POSITIVO verso l'alto (polo superiore/spalla) e NEGATIVO verso il basso (solco sottomammario)
        zc_rel = mlo_nipple_y - zc_interp

        # Conversione in millimetri
        w_mm = w_interp * self.pixel_spacing_mm
        h_mm = h_interp * self.pixel_spacing_mm
        xc_mm = xc_rel * self.pixel_spacing_mm
        zc_mm = zc_rel * self.pixel_spacing_mm

        # Smoothing gaussiano per garantire superfici organiche C1 continue
        w_mm = gaussian_filter1d(w_mm, sigma=2.5)
        h_mm = gaussian_filter1d(h_mm, sigma=2.5)
        xc_mm = gaussian_filter1d(xc_mm, sigma=2.5)
        zc_mm = gaussian_filter1d(zc_mm, sigma=2.5)

        # Convergenza naturale verso il capezzolo
        global_taper = np.sqrt(np.clip(1.0 - np.power(self.t, 2.2), 0.05, 1.0))

        self.W_mm = np.maximum(w_mm * global_taper, 6.0)
        self.H_mm = np.maximum(h_mm * global_taper, 6.0)
        self.Xc_mm = xc_mm * (1.0 - np.power(self.t, 2.0))
        self.Zc_mm = zc_mm * (1.0 - np.power(self.t, 2.0))

    def get_slice_parameters(self, slice_idx: int) -> Tuple[float, float, float, float, float]:
        """
        Ritorna (y_depth, width, height, center_x, center_z) per una data sezione.
        """
        return (
            float(self.depth_y_mm[slice_idx]),
            float(self.W_mm[slice_idx]),
            float(self.H_mm[slice_idx]),
            float(self.Xc_mm[slice_idx]),
            float(self.Zc_mm[slice_idx]),
        )
