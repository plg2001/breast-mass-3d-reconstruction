"""
Modulo di Segmentazione del Profilo Cutaneo (Skin-Line) e PNL
Basato sui principi descritti in:
- Huang et al. (2006) - Reconstruction and rendering by MPGS
- Pinto et al. (2023) - Generative compressed breast shape model
- Rodríguez-Ruiz et al. (2017) - In vivo shape characterization
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


def load_mammogram(image_path: str) -> np.ndarray:
    """Carica l'immagine mammografica in scala di grigi."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossibile caricare l'immagine mammografica: {image_path}")
    return img


def detect_laterality(img: np.ndarray) -> bool:
    """
    Rileva se la parete toracica è a sinistra (True) o a destra (False).
    - Lato sinistro (L): parete toracica a sinistra, capezzolo a destra.
    - Lato destro (R): parete toracica a destra, capezzolo a sinistra.
    """
    h, w = img.shape
    border_w = max(10, int(0.05 * w))
    left_intensity = float(img[:, :border_w].mean())
    right_intensity = float(img[:, -border_w:].mean())
    return left_intensity >= right_intensity


def clean_scanner_artifacts(img: np.ndarray) -> np.ndarray:
    """
    Rimuove artefatti di scansione, strisce di calibrazione e bordi del digitalizzatore.
    Azzera il margine perimetrale esterno del 2.5% per evitare che i bordi del fotogramma
    vengano scambiati per contorni anatomici.
    """
    h, w = img.shape
    cleaned = img.copy()

    margin_y = max(15, int(0.025 * h))
    margin_x = max(15, int(0.025 * w))

    cleaned[:margin_y, :] = 0
    cleaned[h - margin_y:, :] = 0
    cleaned[:, :margin_x] = 0
    cleaned[:, w - margin_x:] = 0

    return cleaned


def segment_breast_mask(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Segmenta la maschera binaria della mammella escludendo il fondo nero e gli artefatti radiologici.
    """
    h, w = img.shape
    target_h = 800
    scale = target_h / float(h)
    target_w = int(w * scale)
    small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(small, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 22, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    if num_labels <= 1:
        raise ValueError("Nessun tessuto mammario rilevato durante la binarizzazione.")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask_small = (labels == largest_label).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(mask_small, cnts, -1, 255, -1)

    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_full, scale


def extract_landmarks_and_profile(
    img: np.ndarray,
    mask: np.ndarray,
    chest_is_left: bool
) -> Dict[str, Any]:
    """
    Estrae l'apice del capezzolo, la linea di parete toracica e il profilo
    della sagoma cutanea lungo l'asse PNL (Posterior Nipple Line).
    Il capezzolo viene ricercato nella fascia anatomica centrale [0.18*H, 0.82*H]
    per evitare artefatti sul bordo superiore (spalla) o inferiore (digitalizzatore).
    """
    h, w = img.shape
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        raise ValueError("Maschera vuota: impossibile estrarre il profilo cutaneo.")

    row_min = int(rows.min())
    row_max = int(rows.max())

    # Fascia di ricerca anatomica per il capezzolo (esclude estremi non fisiologici)
    valid_rows = rows[(rows >= int(0.18 * h)) & (rows <= int(0.82 * h))]
    if len(valid_rows) == 0:
        valid_rows = rows

    if chest_is_left:
        chest_x = 0
        skin_x_per_row = np.array([np.max(np.where(mask[r, :] > 0)[0]) for r in valid_rows])
        apex_idx = int(np.argmax(skin_x_per_row))
        nipple_x = int(skin_x_per_row[apex_idx])
        nipple_y = int(valid_rows[apex_idx])
        pnl_length = float(nipple_x - chest_x)
    else:
        chest_x = w - 1
        skin_x_per_row = np.array([np.min(np.where(mask[r, :] > 0)[0]) for r in valid_rows])
        apex_idx = int(np.argmin(skin_x_per_row))
        nipple_x = int(skin_x_per_row[apex_idx])
        nipple_y = int(valid_rows[apex_idx])
        pnl_length = float(chest_x - nipple_x)

    num_samples = 64
    depth_ratios = np.linspace(0.01, 0.985, num_samples)
    widths = []
    centers = []

    for ratio in depth_ratios:
        if chest_is_left:
            target_x = int(chest_x + ratio * pnl_length)
        else:
            target_x = int(chest_x - ratio * pnl_length)

        target_x = np.clip(target_x, 0, w - 1)
        col_pixels = np.where(mask[:, target_x] > 0)[0]

        if len(col_pixels) > 0:
            y_span_min = float(col_pixels.min())
            y_span_max = float(col_pixels.max())
            span = max(2.0, y_span_max - y_span_min)
            center = 0.5 * (y_span_max + y_span_min)
        else:
            span = 2.0
            center = float(nipple_y)

        widths.append(span)
        centers.append(center)

    widths = np.array(widths)
    centers = np.array(centers)

    for i in range(1, len(widths) - 1):
        if widths[i] <= 2.0:
            widths[i] = 0.5 * (widths[i - 1] + widths[i + 1])
            centers[i] = 0.5 * (centers[i - 1] + centers[i + 1])

    return {
        "chest_is_left": chest_is_left,
        "chest_x": chest_x,
        "nipple": (nipple_x, nipple_y),
        "pnl_length": pnl_length,
        "row_bounds": (row_min, row_max),
        "depth_ratios": depth_ratios,
        "widths": widths,
        "centers": centers,
        "image_shape": (h, w),
    }


def process_mammogram_view(image_path: str) -> Dict[str, Any]:
    """
    Pipeline completa di analisi e segmentazione per una singola vista mammografica.
    """
    raw_img = load_mammogram(image_path)
    chest_is_left = detect_laterality(raw_img)
    cleaned_img = clean_scanner_artifacts(raw_img)
    mask, scale = segment_breast_mask(cleaned_img)
    profile_data = extract_landmarks_and_profile(cleaned_img, mask, chest_is_left)
    profile_data["mask"] = mask
    profile_data["cleaned_image"] = cleaned_img
    profile_data["image_path"] = image_path
    return profile_data
