"""
Input loading and validation.

Ensures that the four input images (R-CC, R-MLO, R-ROI-CC, R-ROI-MLO)
are consistent, correctly paired, and suitable for reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import Config
from .utils import load_image, log


# ── Data container ──────────────────────────────────────────────────────

@dataclass
class InputData:
    """Validated set of input images, all normalised to float64 [0, 1]."""

    cc: np.ndarray         # R-CC mammogram  (H_cc, W_cc)
    mlo: np.ndarray        # R-MLO mammogram (H_mlo, W_mlo)
    roi_cc: np.ndarray     # R-ROI-CC mask   (H_cc, W_cc)  – binary
    roi_mlo: np.ndarray    # R-ROI-MLO mask  (H_mlo, W_mlo) – binary

    @property
    def cc_shape(self) -> Tuple[int, int]:
        return self.cc.shape[:2]

    @property
    def mlo_shape(self) -> Tuple[int, int]:
        return self.mlo.shape[:2]


# ── Validation ──────────────────────────────────────────────────────────

def _check_mask_alignment(
    mammo: np.ndarray,
    mask: np.ndarray,
    mammo_name: str,
    mask_name: str,
) -> None:
    """Verify that a ROI mask matches its mammogram in size."""
    if mammo.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Dimension mismatch: {mammo_name} has shape {mammo.shape[:2]} "
            f"but {mask_name} has shape {mask.shape[:2]}.  "
            "The ROI mask must have the same pixel dimensions as its "
            "corresponding mammogram."
        )


def _check_mask_content(mask: np.ndarray, name: str) -> None:
    """Verify that a mask is roughly binary and non-trivial."""
    unique = np.unique(mask)

    # Allow small floating-point artefacts
    if len(unique) < 2:
        # Might be all-zero or all-one
        if mask.max() < 0.05:
            raise ValueError(
                f"ROI mask '{name}' appears to be empty (all black).  "
                "Provide a mask with at least one white (lesion) region."
            )
        log.warning(
            "ROI mask '%s' has a single intensity level (%.3f). "
            "It may cover the entire image.",
            name,
            float(unique[0]),
        )
        return

    # Check that there's a meaningful foreground region
    fg_frac = (mask > 0.5).sum() / mask.size
    if fg_frac < 1e-6:
        raise ValueError(
            f"ROI mask '{name}' foreground is < 0.0001 %% of the image. "
            "Check that the mask is not empty."
        )
    if fg_frac > 0.90:
        log.warning(
            "ROI mask '%s' foreground covers %.1f %% of the image — "
            "are you sure this is correct?",
            name,
            fg_frac * 100,
        )


def _check_roi_within_breast(
    mammo: np.ndarray,
    mask: np.ndarray,
    mammo_name: str,
    mask_name: str,
) -> None:
    """Warn if the ROI overlaps mostly background."""
    breast_region = mammo > 0.05  # rough breast region
    roi_region = mask > 0.5
    if roi_region.sum() == 0:
        return
    overlap = np.logical_and(roi_region, breast_region).sum()
    frac = overlap / roi_region.sum()
    if frac < 0.5:
        log.warning(
            "Only %.0f %% of '%s' overlaps the breast region in '%s'. "
            "The ROI may not be correctly aligned.",
            frac * 100,
            mask_name,
            mammo_name,
        )


# ── Public entry point ─────────────────────────────────────────────────

def load_and_validate(cfg: Config) -> InputData:
    """Load the four input images and run all consistency checks.

    Returns an InputData instance with normalised float64 images.
    """
    cfg.validate_paths()

    log.info("Loading images …")
    cc = load_image(cfg.r_cc_path, grayscale=True)
    mlo = load_image(cfg.r_mlo_path, grayscale=True)
    roi_cc = load_image(cfg.r_roi_cc_path, grayscale=True)
    roi_mlo = load_image(cfg.r_roi_mlo_path, grayscale=True)

    log.info("  R-CC      : %s", cc.shape)
    log.info("  R-MLO     : %s", mlo.shape)
    log.info("  R-ROI-CC  : %s", roi_cc.shape)
    log.info("  R-ROI-MLO : %s", roi_mlo.shape)

    # ── Dimension checks ───────────────────────────────────────────────
    _check_mask_alignment(cc, roi_cc, "R-CC", "R-ROI-CC")
    _check_mask_alignment(mlo, roi_mlo, "R-MLO", "R-ROI-MLO")

    # ── Mask content checks ────────────────────────────────────────────
    _check_mask_content(roi_cc, "R-ROI-CC")
    _check_mask_content(roi_mlo, "R-ROI-MLO")

    # Binarise masks (threshold at 0.5)
    roi_cc = (roi_cc > 0.5).astype(np.float64)
    roi_mlo = (roi_mlo > 0.5).astype(np.float64)

    # ── ROI–mammogram correspondence ───────────────────────────────────
    _check_roi_within_breast(cc, roi_cc, "R-CC", "R-ROI-CC")
    _check_roi_within_breast(mlo, roi_mlo, "R-MLO", "R-ROI-MLO")

    log.info("Input validation passed ✓")
    return InputData(cc=cc, mlo=mlo, roi_cc=roi_cc, roi_mlo=roi_mlo)
