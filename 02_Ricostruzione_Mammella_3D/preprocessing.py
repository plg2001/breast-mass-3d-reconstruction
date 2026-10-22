"""
Image preprocessing for the 3D reconstruction pipeline.

Responsibilities:
  1. Resize images to working resolution.
  2. Segment the breast silhouette from each mammogram.
  3. Clean ROI masks.
  4. Record scale factors so results can be mapped back to original
     pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage

from .config import Config
from .input_validation import InputData
from .utils import resize_image, bounding_box, log


# ── Data container ──────────────────────────────────────────────────────

@dataclass
class PreprocessedData:
    """Images after resizing, segmentation, and mask cleaning."""

    # Resized mammograms (float64, [0, 1])
    cc: np.ndarray
    mlo: np.ndarray

    # Resized ROI masks (binary float64, 0/1)
    roi_cc: np.ndarray
    roi_mlo: np.ndarray

    # Breast silhouette masks (binary float64, 0/1)
    breast_cc: np.ndarray
    breast_mlo: np.ndarray

    # Scale factors (new / original)
    scale_cc: float
    scale_mlo: float

    # Bounding boxes of the breast in the resized images:
    # (col_min, row_min, col_max, row_max)
    bbox_cc: Tuple[int, int, int, int]
    bbox_mlo: Tuple[int, int, int, int]


# ── Breast segmentation ────────────────────────────────────────────────

def _segment_breast(
    img: np.ndarray,
    blur_sigma: float = 5.0,
    min_area_frac: float = 0.05,
) -> np.ndarray:
    """Segment the breast from the background.

    Steps
    -----
    1. Gaussian blur to suppress noise.
    2. Otsu thresholding (via histogram-based search).
    3. Morphological closing to fill gaps.
    4. Hole filling.
    5. Keep the largest connected component.

    Returns a binary mask (float64, 0/1).
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk, label as ski_label
    try:
        from skimage.morphology import closing
    except ImportError:
        from skimage.morphology import binary_closing as closing

    try:
        from skimage.morphology import remove_small_holes
    except ImportError:
        remove_small_holes = None

    # Step 1: Remove scanner / film border artifacts along non-chest-wall edges
    cleaned_img = img.copy()
    h, w = cleaned_img.shape
    # Film edge artifacts frequently appear at row 0..int(h*0.03) or near right boundary
    # In mammograms, chest wall is on col 0. The region at col > w*0.3 at the top or bottom
    # borders should never be connected to bright non-tissue film clips.
    top_margin = max(int(h * 0.03), 4)
    if cleaned_img[:top_margin, int(w * 0.4):].mean() > 0.2:
        cleaned_img[:top_margin, int(w * 0.25):] = 0.0

    # Step 2: blur
    blurred = ndimage.gaussian_filter(cleaned_img, sigma=blur_sigma)

    # Step 3: Otsu threshold
    try:
        thresh = threshold_otsu(blurred)
    except ValueError:
        # Uniform image — fallback
        thresh = 0.1
    mask = blurred > thresh

    # Step 3: morphological closing
    selem = disk(max(3, int(min(img.shape) * 0.01)))
    mask = closing(mask, selem)

    # Step 4: fill holes
    if remove_small_holes is not None:
        try:
            mask = remove_small_holes(mask, max_size=int(mask.size * 0.01))
        except TypeError:
            mask = remove_small_holes(mask, area_threshold=int(mask.size * 0.01))

    # Step 5: largest connected component
    labelled, n = ndimage.label(mask)
    if n == 0:
        log.warning("No breast region found — returning full-image mask.")
        return np.ones_like(img, dtype=np.float64)
    sizes = ndimage.sum(mask, labelled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    mask = (labelled == largest)

    # Sanity check
    area_frac = mask.sum() / mask.size
    if area_frac < min_area_frac:
        log.warning(
            "Breast mask covers only %.1f %% of the image — segmentation "
            "may be unreliable.",
            area_frac * 100,
        )

    return mask.astype(np.float64)


# ── ROI mask cleanup ───────────────────────────────────────────────────

def _clean_roi(mask: np.ndarray) -> np.ndarray:
    """Clean a binary ROI mask.

    - Remove small speckle components.
    - Keep the largest connected component (the main lesion).
    """
    binary = mask > 0.5
    labelled, n = ndimage.label(binary)
    if n == 0:
        return mask
    if n == 1:
        return binary.astype(np.float64)
    sizes = ndimage.sum(binary, labelled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return (labelled == largest).astype(np.float64)


# ── Public entry point ─────────────────────────────────────────────────

def preprocess(cfg: Config, data: InputData) -> PreprocessedData:
    """Resize images, segment breast, clean ROI masks."""

    ws = cfg.working_size
    log.info("Preprocessing (working size = %d) …", ws)

    # Resize mammograms (bilinear)
    cc, scale_cc = resize_image(data.cc, ws, order=1)
    mlo, scale_mlo = resize_image(data.mlo, ws, order=1)
    log.info("  CC  resized to %s  (scale %.4f)", cc.shape, scale_cc)
    log.info("  MLO resized to %s  (scale %.4f)", mlo.shape, scale_mlo)

    # Resize ROI masks (nearest-neighbour to keep binary)
    roi_cc, _ = resize_image(data.roi_cc, ws, order=0)
    roi_mlo, _ = resize_image(data.roi_mlo, ws, order=0)
    roi_cc = _clean_roi(roi_cc)
    roi_mlo = _clean_roi(roi_mlo)

    # Segment breast silhouettes
    breast_cc = _segment_breast(cc, cfg.breast_blur_sigma, cfg.breast_min_area_frac)
    breast_mlo = _segment_breast(mlo, cfg.breast_blur_sigma, cfg.breast_min_area_frac)
    log.info(
        "  Breast mask CC  : %.1f %% foreground",
        breast_cc.mean() * 100,
    )
    log.info(
        "  Breast mask MLO : %.1f %% foreground",
        breast_mlo.mean() * 100,
    )

    # Bounding boxes of breast silhouettes
    bbox_cc = bounding_box(breast_cc)
    bbox_mlo = bounding_box(breast_mlo)
    log.info("  Breast bbox CC  : %s", bbox_cc)
    log.info("  Breast bbox MLO : %s", bbox_mlo)

    return PreprocessedData(
        cc=cc,
        mlo=mlo,
        roi_cc=roi_cc,
        roi_mlo=roi_mlo,
        breast_cc=breast_cc,
        breast_mlo=breast_mlo,
        scale_cc=scale_cc,
        scale_mlo=scale_mlo,
        bbox_cc=bbox_cc,
        bbox_mlo=bbox_mlo,
    )
