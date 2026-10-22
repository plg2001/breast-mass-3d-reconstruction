"""
End-to-end test with synthetic images.

Creates synthetic mammogram and ROI images based on clinical projection
geometry, runs the full pipeline, and verifies that outputs are produced correctly.

Usage:
    python -m mammography_3d.test_synthetic
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def create_synthetic_images(out_dir: Path) -> dict:
    """Create synthetic mammogram and ROI mask images.

    Coordinate system:
      - Column (horizontal): Depth (chest wall at col 0, nipple towards the right).
      - Row (vertical): Medial-Lateral in CC, oblique in MLO.
    """
    h, w = 256, 200  # rows=256, cols=200

    # Grid of pixel coordinates: xx is column (0..w-1), yy is row (0..h-1)
    yy, xx = np.mgrid[0:h, 0:w]

    # ── Breast: half-ellipse anchored at chest wall (col 0) ───────────
    # In CC:
    # Chest wall is at col 0. Breast extends to col ~160. Centred vertically at row 128.
    r_depth = 160.0
    r_height = 100.0
    y_center = 128.0
    breast_cc = ((xx / r_depth) ** 2 + ((yy - y_center) / r_height) ** 2 <= 1.0) & (xx >= 0)

    cc = np.zeros((h, w), dtype=np.uint8)
    cc[breast_cc] = 180
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 10, (h, w))
    cc = np.clip(cc.astype(float) + noise * breast_cc, 0, 255).astype(np.uint8)

    # In MLO:
    # Similar breast shape with slight tilt
    breast_mlo = ((xx / (r_depth * 1.05)) ** 2 + ((yy - y_center) / (r_height * 1.05)) ** 2 <= 1.0) & (xx >= 0)
    mlo = np.zeros((h, w), dtype=np.uint8)
    mlo[breast_mlo] = 175
    mlo = np.clip(mlo.astype(float) + noise * breast_mlo, 0, 255).astype(np.uint8)

    # ── 3D Lesion Position ─────────────────────────────────────────────
    # Choose 3D coordinates: x0 (row in CC), y0 (depth/col), z0 (height)
    x0 = 110.0   # row in CC
    y0 = 60.0    # col in CC (depth from chest wall)
    z0 = 30.0    # height
    theta = np.radians(45.0)

    # CC projection: col = y0, row = x0
    col_cc_lesion = y0
    row_cc_lesion = x0

    # MLO projection: col = y0, row = z0 * cos(theta) + x0 * sin(theta)
    col_mlo_lesion = y0
    row_mlo_lesion = z0 * np.cos(theta) + x0 * np.sin(theta)

    r_lesion = 12.0

    # ROI CC
    roi_cc = np.zeros((h, w), dtype=np.uint8)
    mask_cc = ((xx - col_cc_lesion) ** 2 + (yy - row_cc_lesion) ** 2) <= r_lesion ** 2
    roi_cc[mask_cc] = 255

    # ROI MLO
    roi_mlo = np.zeros((h, w), dtype=np.uint8)
    mask_mlo = ((xx - col_mlo_lesion) ** 2 + (yy - row_mlo_lesion) ** 2) <= r_lesion ** 2
    roi_mlo[mask_mlo] = 255

    # ── Save ───────────────────────────────────────────────────────────
    paths = {}
    for name, img in [("R_CC", cc), ("R_MLO", mlo),
                      ("R_ROI_CC", roi_cc), ("R_ROI_MLO", roi_mlo)]:
        p = out_dir / f"{name}.png"
        Image.fromarray(img).save(str(p))
        paths[name] = str(p)

    return paths


def run_test():
    """Run the full pipeline on synthetic data."""
    from mammography_3d.config import Config
    from mammography_3d.main import run_pipeline

    with tempfile.TemporaryDirectory(prefix="mammo3d_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Test directory: {tmpdir}")

        paths = create_synthetic_images(tmpdir)
        print("Synthetic images created.")

        output_dir = tmpdir / "output"
        cfg = Config(
            r_cc_path=paths["R_CC"],
            r_mlo_path=paths["R_MLO"],
            r_roi_cc_path=paths["R_ROI_CC"],
            r_roi_mlo_path=paths["R_ROI_MLO"],
            output_dir=str(output_dir),
            voxel_resolution=100,
            working_size=256,
        )

        run_pipeline(cfg)

        expected_files = [
            "config.json",
            "validation_report.json",
            "validation_report.txt",
            "meshes/breast.glb",
            "meshes/glandular.glb",
            "meshes/pectoral.glb",
            "meshes/ducts.glb",
            "meshes/lesion.glb",
            "meshes/combined_scene.glb",
            "figures/01_input_overview.png",
            "figures/02_breast_silhouette_validation.png",
            "figures/03_lesion_roi_validation.png",
            "figures/05_validation_summary.png",
            "figures/06_internal_tissues_3d.png",
        ]

        all_ok = True
        for rel in expected_files:
            p = output_dir / rel
            if p.exists():
                print(f"  [OK] {rel}")
            else:
                print(f"  [FAIL] {rel} -- MISSING")
                all_ok = False

        if all_ok:
            print("\n[SUCCESS] All expected outputs generated successfully!")
        else:
            print("\n[ERROR] Some outputs are missing.")
            sys.exit(1)


if __name__ == "__main__":
    run_test()
