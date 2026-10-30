"""
Dataset e preprocessing per mammografie 2D -> 3D.
"""

import random
import numpy as np
import cv2
import scipy.ndimage as ndimage
import torch
from torch.utils.data import Dataset

from config import Config


def build_seed_and_target_volumes(img_2d: np.ndarray):
    """
    Implementa la Sez. IV-A (Eq. 1):
    Dato un array 2D normalizzato [128, 128] in [0, 1], genera:
    - seed_vol V^(0): [1, 10, 128, 128] con decadimento lungo z (Eq. 1)
    - target_vol V  : [1, 10, 128, 128] volume 3D coerente con profilo anatomico
    """
    H, W = img_2d.shape
    D = Config.NUM_SLICES

    # 1. Seed Volume V^(0) secondo Eq. (1): z_i = (i - 1) * Delta z
    seed_slices = []
    for i in range(D):
        z_i = i * Config.SLICE_INTERVAL
        decay_factor = np.exp(-Config.ALPHA_DECAY * (z_i / Config.SLICE_INTERVAL))
        slice_i = img_2d * decay_factor
        seed_slices.append(slice_i)

    seed_vol = np.stack(seed_slices, axis=0).astype(np.float32)  # [10, 128, 128]
    s_min, s_max = seed_vol.min(), seed_vol.max()
    if s_max - s_min > 1e-6:
        seed_vol = (seed_vol - s_min) / (s_max - s_min)

    # 2. Target Volume V (Riferimento 3D)
    z_coords = np.linspace(-1.0, 1.0, D)[:, None, None]  # [10, 1, 1]
    local_thickness = np.clip(img_2d[None, :, :] * 1.15, 0.15, 1.0)
    envelope_3d = np.exp(-0.5 * (z_coords / local_thickness) ** 2)

    target_vol = (img_2d[None, :, :] * envelope_3d).astype(np.float32)
    target_vol = ndimage.gaussian_filter(target_vol, sigma=(0.6, 0.5, 0.5))
    t_min, t_max = target_vol.min(), target_vol.max()
    if t_max - t_min > 1e-6:
        target_vol = (target_vol - t_min) / (t_max - t_min)

    return seed_vol[None, ...], target_vol[None, ...]


def generate_synthetic_mammogram(size=128):
    """
    Genera una mammografia 2D sintetica realistica (fallback senza dataset).
    """
    y, x = np.ogrid[:size, :size]
    cx, cy = 0, size // 2
    rx, ry = int(size * 0.72), int(size * 0.45)
    breast_mask = ((x - cx)**2 / (rx**2 + 1e-5) + (y - cy)**2 / (ry**2 + 1e-5)) <= 1.0
    dist_from_chest = np.clip(1.0 - (x / (rx + 1e-5)) * 0.55, 0.0, 1.0)

    noise1 = ndimage.gaussian_filter(np.random.rand(size, size), sigma=3.5)
    noise2 = ndimage.gaussian_filter(np.random.rand(size, size), sigma=1.5)
    parenchyma = 0.45 * noise1 + 0.25 * noise2
    parenchyma = (parenchyma - parenchyma.min()) / (parenchyma.max() - parenchyma.min() + 1e-6)

    img = breast_mask * dist_from_chest * (0.20 + 0.48 * parenchyma)

    num_lesions = random.choice([1, 2])
    for _ in range(num_lesions):
        lx = random.randint(int(size * 0.18), int(size * 0.48))
        ly = random.randint(int(size * 0.35), int(size * 0.65))
        lr = random.uniform(4.0, 9.0)
        lesion_profile = np.exp(-((x - lx)**2 + (y - ly)**2) / (2 * (lr**2)))
        img = np.maximum(img, breast_mask * lesion_profile * random.uniform(0.82, 0.98))

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return (img * 255).astype(np.uint8)


class Mammogram2DTo3DDataset(Dataset):
    """
    Dataset per la conversione 2D -> 3D di mammografie.
    """
    def __init__(self, image_paths, augment=False):
        self.image_paths = image_paths
        self.augment = augment
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.image_paths)

    def _preprocess_2d(self, img_gray: np.ndarray) -> np.ndarray:
        """Preprocessing della mammografia 2D."""
        p1, p99 = np.percentile(img_gray, (1, 99))
        if p99 > p1:
            img_gray = np.clip(img_gray, p1, p99)
            img_gray = ((img_gray - p1) / (p99 - p1) * 255.0).astype(np.uint8)

        img_resized = cv2.resize(img_gray, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA)

        if self.augment:
            if random.random() < 0.5:
                img_resized = cv2.flip(img_resized, 1)
            if random.random() < 0.3:
                img_resized = self.clahe.apply(img_resized)
            angle = random.uniform(-5.0, 5.0)
            scale = random.uniform(0.95, 1.05)
            tx = random.uniform(-0.05, 0.05) * Config.IMG_SIZE
            ty = random.uniform(-0.05, 0.05) * Config.IMG_SIZE
            M = cv2.getRotationMatrix2D((Config.IMG_SIZE / 2, Config.IMG_SIZE / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img_resized = cv2.warpAffine(img_resized, M, (Config.IMG_SIZE, Config.IMG_SIZE),
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        img_float = img_resized.astype(np.float32) / 255.0

        if self.augment:
            gamma = random.uniform(0.9, 1.1)
            img_float = np.power(np.clip(img_float, 1e-6, 1.0), gamma)
            sigma = random.uniform(0.0, 0.01)
            img_float = img_float + np.random.normal(0, sigma, img_float.shape).astype(np.float32)

        imin, imax = img_float.min(), img_float.max()
        if imax - imin > 1e-6:
            img_float = (img_float - imin) / (imax - imin)
        return np.clip(img_float, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if path.startswith("SYNTHETIC_"):
            raw_img = generate_synthetic_mammogram(Config.IMG_SIZE)
        else:
            raw_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                raw_img = generate_synthetic_mammogram(Config.IMG_SIZE)

        img_2d = self._preprocess_2d(raw_img)
        seed_vol, target_vol = build_seed_and_target_volumes(img_2d)
        return torch.from_numpy(seed_vol), torch.from_numpy(target_vol), torch.from_numpy(img_2d)
