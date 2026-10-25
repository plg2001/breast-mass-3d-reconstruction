#!/usr/bin/env python3
"""
Genera un solo GLB con una superficie mammellare esterna liscia e senza capezzolo,
partendo dalle proiezioni CC e MLO.

Dipendenze:
    pip install numpy opencv-python scipy trimesh

Esempio:
    python breast_surface_glb_v2.py --cc R-CC.jpg --mlo R-MLO.jpg --output breast_surface.glb

IMPORTANTE:
La mesh è una superficie plausibile ricavata da profili 2D semplificati; CC e MLO
non determinano da sole una forma 3D univoca. Verificare sempre la segmentazione.
Questo script esporta esclusivamente il GLB finale.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d, binary_fill_holes


def load_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File non trovato: {p}")
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Impossibile leggere l'immagine: {p}")
    return im


def largest_component(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n < 2:
        raise ValueError("Nessuna regione foreground identificata.")
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == idx


def extract_profile(image: np.ndarray, max_side: int = 1200) -> dict:
    """Segmentazione euristica e profili superiore/inferiore della silhouette."""
    h0, w0 = image.shape
    scale = min(1.0, max_side / max(h0, w0))
    im = cv2.resize(image, (round(w0 * scale), round(h0 * scale)),
                    interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()

    # Contrasto robusto per immagini mammografiche chiare o scure.
    p2, p98 = np.percentile(im, [2, 98])
    if p98 <= p2:
        raise ValueError("Immagine priva di contrasto.")
    norm = np.clip((im.astype(np.float32) - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    threshold = max(8, int(np.percentile(blur[blur > 0], 20) * 0.38)) if np.any(blur > 0) else 8
    binary = (blur > threshold).astype(np.uint8)

    # Esclude il bordo del digitalizzatore senza tagliare una porzione eccessiva.
    border = max(2, int(round(min(binary.shape) * 0.004)))
    binary[:border] = 0
    binary[-border:] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    k = max(3, int(round(min(binary.shape) * 0.009)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = binary_fill_holes(largest_component(binary))

    ys, xs = np.where(mask)
    if len(xs) < 100:
        raise ValueError("Contorno mammario non identificato. Controlla input/contrasto.")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape

    top = np.full(w, np.nan)
    bottom = np.full(w, np.nan)
    for x in range(w):
        rows = np.flatnonzero(crop[:, x])
        if rows.size:
            top[x], bottom[x] = rows[0], rows[-1]

    good = np.isfinite(top) & np.isfinite(bottom)
    if good.sum() < max(10, w // 8):
        raise ValueError("Profilo troppo discontinuo per costruire una mesh.")
    xx = np.arange(w)
    top = np.interp(xx, xx[good], top[good])
    bottom = np.interp(xx, xx[good], bottom[good])

    sigma = max(2.0, w * 0.012)
    top = gaussian_filter1d(top, sigma=sigma)
    bottom = gaussian_filter1d(bottom, sigma=sigma)
    thickness = np.maximum(bottom - top, 1.0)
    center = (top + bottom) * 0.5

    # La parete toracica tende a trovarsi sul lato con maggiore occupazione
    # nella zona laterale; normalizziamo la coordinata longitudinale.
    q = max(1, w // 5)
    chest_right = crop[:, -q:].mean() > crop[:, :q].mean()
    if chest_right:
        thickness = thickness[::-1]
        center = center[::-1]

    return {
        "length_px": float(max(w - 1, 1)),
        "thickness": thickness,
        "center": center,
    }


def resample(a: np.ndarray, n: int) -> np.ndarray:
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)


def make_mesh(cc: dict, mlo: dict, depth_mm: float, slices: int, radial: int,
              width_factor: float, height_factor: float) -> trimesh.Trimesh:
    t = np.linspace(0, 1, slices)
    cc_th = resample(cc["thickness"], slices)
    ml_th = resample(mlo["thickness"], slices)
    cc_ctr = resample(cc["center"], slices)
    ml_ctr = resample(mlo["center"], slices)

    sx = depth_mm / cc["length_px"]
    sz = depth_mm / mlo["length_px"]

    # Sezioni con rastremazione dolce: niente punta/capezzolo.
    # L'ampiezza non collassa a zero alle estremità; una calotta sferica
    # chiude il volume e riduce l'effetto di taglio piatto.
    taper = np.sin(np.pi * (0.025 + 0.95 * t)) ** 0.48
    taper = np.clip(taper, 0.16, 1.0)

    width = np.maximum(5.0, cc_th * sx * width_factor) * taper
    height = np.maximum(5.0, ml_th * sz * height_factor) * taper
    xc = gaussian_filter1d((cc_ctr - np.median(cc_ctr)) * sx * 0.35, max(1, slices * .025))
    zc = gaussian_filter1d((np.median(ml_ctr) - ml_ctr) * sz * 0.35, max(1, slices * .025))
    width = gaussian_filter1d(width, max(1, slices * .018))
    height = gaussian_filter1d(height, max(1, slices * .018))

    theta = np.linspace(0, 2 * np.pi, radial, endpoint=False)
    p = 2.25
    ct = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2 / p)
    st = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2 / p)

    verts = []
    for i, ti in enumerate(t):
        # Raccordo leggermente appiattito verso la parete toracica.
        posterior = 1.0 - 0.08 * (1 - ti) ** 2
        for j in range(radial):
            verts.append([
                xc[i] + 0.5 * width[i] * ct[j] * posterior,
                ti * depth_mm,
                zc[i] + 0.5 * height[i] * st[j] * posterior,
            ])
    verts = np.asarray(verts, dtype=np.float64)

    faces = []
    for i in range(slices - 1):
        a0, a1 = i * radial, (i + 1) * radial
        for j in range(radial):
            jn = (j + 1) % radial
            # Indici ordinati in modo coerente; fix_normals corregge l'orientamento.
            faces.extend(([a0 + j, a0 + jn, a1 + jn],
                          [a0 + j, a1 + jn, a1 + j]))

    # Tappi con vertici centrali spostati appena all'interno per evitare un bordo vivo.
    i0 = len(verts)
    i1 = i0 + 1
    verts = np.vstack([verts, [xc[0], 0.0, zc[0]], [xc[-1], depth_mm, zc[-1]]])
    last = (slices - 1) * radial
    for j in range(radial):
        jn = (j + 1) % radial
        faces.append([i0, jn, j])
        faces.append([last + j, last + jn, i1])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.visual.face_colors = np.tile(np.array([224, 220, 214, 255], dtype=np.uint8),
                                      (len(mesh.faces), 1))
    return mesh


def main():
    ap = argparse.ArgumentParser(description="Esporta la superficie mammellare in un unico GLB.")
    ap.add_argument("--cc", required=True, help="Mammografia CC")
    ap.add_argument("--mlo", required=True, help="Mammografia MLO")
    ap.add_argument("--output", default="breast_surface.glb", help="Percorso GLB finale")
    ap.add_argument("--depth-mm", type=float, default=150.0)
    ap.add_argument("--slices", type=int, default=120)
    ap.add_argument("--radial-segments", type=int, default=112)
    ap.add_argument("--width-scale", type=float, default=1.0)
    ap.add_argument("--height-scale", type=float, default=1.0)
    args = ap.parse_args()

    if args.depth_mm <= 0 or args.slices < 16 or args.radial_segments < 16:
        ap.error("depth-mm deve essere >0 e slices/radial-segments almeno 16.")

    cc = extract_profile(load_image(args.cc))
    mlo = extract_profile(load_image(args.mlo))
    mesh = make_mesh(cc, mlo, args.depth_mm, args.slices, args.radial_segments,
                     args.width_scale, args.height_scale)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out), file_type="glb")
    print(f"GLB creato: {out.resolve()}")


if __name__ == "__main__":
    main()
