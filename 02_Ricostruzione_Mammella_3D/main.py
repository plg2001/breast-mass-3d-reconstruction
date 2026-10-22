"""
Main pipeline for 3-D mammography reconstruction.

Usage
─────
    python -m mammography_3d.main \
        --r-cc      path/to/R_CC.png \
        --r-mlo     path/to/R_MLO.png \
        --r-roi-cc  path/to/R_ROI_CC.png \
        --r-roi-mlo path/to/R_ROI_MLO.png \
        --output    output/

All parameters listed in config.py can also be passed as CLI arguments
(replace underscores with hyphens).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config
from .input_validation import load_and_validate
from .preprocessing import preprocess
from .geometry import ProjectionGeometry
from .breast_reconstruction import reconstruct_breast
from .lesion_reconstruction import reconstruct_lesion
from .internal_tissues import reconstruct_internal_tissues
from .validation import validate
from .visualization import generate_all_figures
from .export_meshes import export_all
from .utils import log


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct a 3-D breast + lesion model from "
                    "CC and MLO mammographic views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs
    p.add_argument("--r-cc", required=True, help="Path to R-CC mammogram.")
    p.add_argument("--r-mlo", required=True, help="Path to R-MLO mammogram.")
    p.add_argument("--r-roi-cc", required=True, help="Path to R-ROI-CC mask.")
    p.add_argument("--r-roi-mlo", required=True, help="Path to R-ROI-MLO mask.")

    # Output
    p.add_argument("--output", default="output", help="Output directory.")

    # Mode
    p.add_argument("--mode", choices=["anatomical", "visual_hull"], default="anatomical",
                    help="Reconstruction mode: 'anatomical' (smooth, organic patient-specific dome) or 'visual_hull' (raw prism).")

    # Geometry
    p.add_argument("--mlo-angle", type=float, default=45.0,
                    help="MLO angle in degrees from horizontal.")
    p.add_argument("--voxel-resolution", type=int, default=200,
                    help="Voxels along the longest grid axis.")
    p.add_argument("--working-size", type=int, default=512,
                    help="Max image dimension for processing.")

    # Smoothing
    p.add_argument("--breast-volume-sigma", type=float, default=2.0,
                    help="Gaussian sigma for breast volume smoothing.")
    p.add_argument("--lesion-volume-sigma", type=float, default=1.5,
                    help="Gaussian sigma for lesion volume smoothing.")
    p.add_argument("--breast-smooth-iter", type=int, default=30,
                    help="Laplacian smoothing iterations (breast).")
    p.add_argument("--lesion-smooth-iter", type=int, default=20,
                    help="Laplacian smoothing iterations (lesion).")

    # Decimation
    p.add_argument("--max-faces-breast", type=int, default=50000,
                    help="Max faces for breast mesh (0 = no limit).")
    p.add_argument("--max-faces-lesion", type=int, default=20000,
                    help="Max faces for lesion mesh (0 = no limit).")

    # Flags
    p.add_argument("--no-validation", action="store_true",
                    help="Skip validation step.")
    p.add_argument("--no-3d-vis", action="store_true",
                    help="Skip 3-D visualisation (faster).")

    return p


def cli_to_config(args: argparse.Namespace) -> Config:
    """Map CLI args to a Config instance."""
    return Config(
        r_cc_path=args.r_cc,
        r_mlo_path=args.r_mlo,
        r_roi_cc_path=args.r_roi_cc,
        r_roi_mlo_path=args.r_roi_mlo,
        output_dir=args.output,
        reconstruction_mode=args.mode,
        mlo_angle_deg=args.mlo_angle,
        voxel_resolution=args.voxel_resolution,
        working_size=args.working_size,
        breast_volume_sigma=args.breast_volume_sigma,
        lesion_volume_sigma=args.lesion_volume_sigma,
        breast_smooth_iterations=args.breast_smooth_iter,
        lesion_smooth_iterations=args.lesion_smooth_iter,
        max_faces_breast=args.max_faces_breast,
        max_faces_lesion=args.max_faces_lesion,
        run_validation=not args.no_validation,
        run_3d_visualisation=not args.no_3d_vis,
    )


def run_pipeline(cfg: Config) -> None:
    """Execute the full reconstruction pipeline."""

    t0 = time.time()
    log.info("═══════════════════════════════════════════════")
    log.info("  Mammography 3-D Reconstruction Pipeline")
    log.info("═══════════════════════════════════════════════")

    # ── 1. Input validation ────────────────────────────────────────────
    log.info("")
    log.info("STEP 1/7 — Input Validation")
    data = load_and_validate(cfg)

    # ── 2. Preprocessing ──────────────────────────────────────────────
    log.info("")
    log.info("STEP 2/7 — Preprocessing")
    prep = preprocess(cfg, data)

    # ── 3. Geometry setup ─────────────────────────────────────────────
    log.info("")
    log.info("STEP 3/7 — Geometry Setup")
    geom = ProjectionGeometry(mlo_angle_deg=cfg.mlo_angle_deg)
    log.info("  MLO angle = %.1f° (cos=%.4f, sin=%.4f)",
             cfg.mlo_angle_deg, geom.cos_theta, geom.sin_theta)

    # ── 4. Breast reconstruction ──────────────────────────────────────
    log.info("")
    log.info("STEP 4/8 — Breast Reconstruction")
    breast_mesh, grid, breast_hull = reconstruct_breast(cfg, prep, geom)

    # ── 5. Lesion reconstruction ──────────────────────────────────────
    log.info("")
    log.info("STEP 5/8 — Lesion Reconstruction")
    lesion_mesh, lesion_hull, lesion_centroid = reconstruct_lesion(
        cfg, prep, geom, grid, breast_hull
    )

    # ── 6. Internal tissues reconstruction ────────────────────────────
    log.info("")
    log.info("STEP 6/8 — Internal Tissues Reconstruction")
    internal_tissues = reconstruct_internal_tissues(cfg, prep, geom, breast_mesh)

    # ── 7. Validation ─────────────────────────────────────────────────
    metrics = None
    if cfg.run_validation:
        log.info("")
        log.info("STEP 7/8 — Validation")
        metrics = validate(
            cfg, prep, geom, grid,
            breast_hull, lesion_hull,
            breast_mesh, lesion_mesh,
            lesion_centroid,
        )
        log.info("\n%s", metrics.summary())

        # Save metrics
        metrics_path = cfg.output_path("validation_report.json")
        with open(str(metrics_path), "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False,
                      default=str)
        log.info("  Metrics saved to %s", metrics_path.name)

        # Save text report
        report_path = cfg.output_path("validation_report.txt")
        with open(str(report_path), "w", encoding="utf-8") as f:
            f.write(metrics.summary())

    # ── 8. Visualisation & Export ─────────────────────────────────────
    log.info("")
    log.info("STEP 8/8 — Visualisation & Export")
    if metrics is not None:
        generate_all_figures(cfg, prep, breast_mesh, lesion_mesh, metrics, internal_tissues)
    export_all(cfg, breast_mesh, lesion_mesh, internal_tissues)

    # Save config
    cfg.save()

    elapsed = time.time() - t0
    log.info("")
    log.info("═══════════════════════════════════════════════")
    log.info("  Pipeline complete in %.1f seconds", elapsed)
    log.info("  Output directory: %s", Path(cfg.output_dir).resolve())
    log.info("═══════════════════════════════════════════════")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = cli_to_config(args)
    try:
        run_pipeline(cfg)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
