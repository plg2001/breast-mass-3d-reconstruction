"""
Utility helpers shared across modules.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# ── Logging ─────────────────────────────────────────────────────────────

def get_logger(name: str = "mammo3d") -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handler.stream = open(sys.stdout.fileno(), mode='w',
                              encoding='utf-8', errors='replace', closefd=False)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


# ── Image I/O ───────────────────────────────────────────────────────────

def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """Load an image from *path* and return it as a NumPy array.

    Supports common raster formats (PNG, TIFF, JPEG, BMP) via Pillow
    and DICOM via pydicom (if installed).

    Parameters
    ----------
    path : str
        Absolute or relative path to the image.
    grayscale : bool
        If True, convert to single-channel float64 in [0, 1].

    Returns
    -------
    np.ndarray
        2-D float64 array with values in [0, 1] (if *grayscale*),
        or 2-D / 3-D uint array otherwise.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")

    suffix = p.suffix.lower()

    if suffix in (".dcm", ".dicom") or suffix == "":
        # Attempt DICOM
        try:
            import pydicom
            ds = pydicom.dcmread(str(p))
            img = ds.pixel_array.astype(np.float64)
        except ImportError:
            raise ImportError(
                f"File '{p.name}' looks like DICOM but pydicom is not "
                "installed.  Run:  pip install pydicom"
            )
    else:
        from PIL import Image
        pil_img = Image.open(str(p))
        if grayscale:
            pil_img = pil_img.convert("L")
        img = np.array(pil_img, dtype=np.float64)

    if grayscale and img.ndim == 3:
        # Convert to grayscale: simple luminance
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    if grayscale:
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)

    return img


def resize_image(
    img: np.ndarray, max_side: int, order: int = 1
) -> Tuple[np.ndarray, float]:
    """Resize *img* so that the longest side equals *max_side*.

    Parameters
    ----------
    img : np.ndarray
        2-D image.
    max_side : int
        Target size for the longest edge.
    order : int
        Interpolation order (0 = nearest, 1 = bilinear, 3 = cubic).

    Returns
    -------
    (resized, scale_factor)
        *scale_factor* is new_size / old_size  (< 1 when shrinking).
    """
    from skimage.transform import resize as ski_resize

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img.copy(), 1.0
    scale = max_side / longest
    new_h = max(int(round(h * scale)), 1)
    new_w = max(int(round(w * scale)), 1)
    resized = ski_resize(
        img, (new_h, new_w), order=order, preserve_range=True, anti_aliasing=True
    )
    return resized.astype(img.dtype), scale


def bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (col_min, row_min, col_max, row_max) of non-zero pixels."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("Mask is empty — no non-zero pixels found.")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(cmin), int(rmin), int(cmax), int(rmax)


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Sørensen–Dice coefficient between two binary masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
