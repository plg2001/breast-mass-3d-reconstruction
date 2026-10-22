"""
Forward-projection validation.

Projects the reconstructed 3-D volumes back onto the CC and MLO
image planes and compares the resulting silhouettes with the original
masks.  Reports Dice coefficient and IoU for both the breast and
the lesion in each view.

Key distinction
───────────────
High projection-match metrics (Dice / IoU) prove that the 3-D model
is *consistent* with the 2-D data.  They do NOT prove that the 3-D
geometry is anatomically correct — infinitely many volumes can produce
the same pair of silhouettes.  This limitation is inherent to having
only two views and is explicitly flagged in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .config import Config
from .geometry import ProjectionGeometry, VoxelGrid
from .preprocessing import PreprocessedData
from .utils import dice_coefficient, iou, log


@dataclass
class ValidationMetrics:
    """Container for all validation results."""

    # Per-view Dice and IoU for the breast silhouette
    breast_dice_cc: float = 0.0
    breast_iou_cc: float = 0.0
    breast_dice_mlo: float = 0.0
    breast_iou_mlo: float = 0.0

    # Per-view Dice and IoU for the lesion ROI
    lesion_dice_cc: float = 0.0
    lesion_iou_cc: float = 0.0
    lesion_dice_mlo: float = 0.0
    lesion_iou_mlo: float = 0.0

    # Mesh quality
    breast_vertices: int = 0
    breast_faces: int = 0
    breast_is_watertight: bool = False
    breast_volume_voxels: int = 0

    lesion_vertices: int = 0
    lesion_faces: int = 0
    lesion_is_watertight: bool = False
    lesion_volume_voxels: int = 0

    # Lesion centroid comparison
    centroid_triangulated: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    centroid_voxel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    centroid_distance: float = 0.0

    # Warnings and notes
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            "===============================================",
            "           VALIDATION REPORT",
            "===============================================",
            "",
            "-- Breast Silhouette Matching --",
            f"  CC  view  |  Dice = {self.breast_dice_cc:.4f}  |  IoU = {self.breast_iou_cc:.4f}",
            f"  MLO view  |  Dice = {self.breast_dice_mlo:.4f}  |  IoU = {self.breast_iou_mlo:.4f}",
            "",
            "-- Lesion ROI Matching --",
            f"  CC  view  |  Dice = {self.lesion_dice_cc:.4f}  |  IoU = {self.lesion_iou_cc:.4f}",
            f"  MLO view  |  Dice = {self.lesion_dice_mlo:.4f}  |  IoU = {self.lesion_iou_mlo:.4f}",
            "",
            "-- Mesh Quality --",
            f"  Breast  |  {self.breast_vertices:,} verts  |  {self.breast_faces:,} faces  |  "
            f"watertight: {self.breast_is_watertight}",
            f"  Lesion  |  {self.lesion_vertices:,} verts  |  {self.lesion_faces:,} faces  |  "
            f"watertight: {self.lesion_is_watertight}",
            "",
            "-- Lesion Centroid --",
            f"  Triangulated : ({self.centroid_triangulated[0]:.1f}, "
            f"{self.centroid_triangulated[1]:.1f}, {self.centroid_triangulated[2]:.1f})",
            f"  Voxel-based  : ({self.centroid_voxel[0]:.1f}, "
            f"{self.centroid_voxel[1]:.1f}, {self.centroid_voxel[2]:.1f})",
            f"  Distance     : {self.centroid_distance:.2f} px",
            "",
        ]
        if self.notes:
            lines.append("-- Notes --")
            for note in self.notes:
                lines.append(f"  * {note}")
            lines.append("")
        lines.extend([
            "-- Interpretation --",
            "  High Dice/IoU values confirm that the 3-D model is",
            "  CONSISTENT with the 2-D input images (its projections",
            "  match the original silhouettes).",
            "",
            "  This does NOT prove anatomical accuracy.  With only two",
            "  views, infinitely many 3-D shapes could produce the same",
            "  pair of silhouettes.  The reconstructed shape represents",
            "  the VISUAL HULL — the maximal volume consistent with the",
            "  observed data.",
            "===============================================",
        ])
        return "\n".join(lines)


# ── Forward projection of voxel volume ─────────────────────────────────

def _forward_project_volume(
    volume: np.ndarray,
    grid: VoxelGrid,
    geom: ProjectionGeometry,
    view: str,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Project a voxel volume onto an image plane.

    Accounts for the discrete voxel footprint via morphological dilation
    and hole-filling so the projection reflects a continuous solid volume.

    Parameters
    ----------
    volume : np.ndarray of bool, shape grid.shape
    grid : VoxelGrid
    geom : ProjectionGeometry
    view : 'cc' or 'mlo'
    target_shape : (H, W) of the target image

    Returns
    -------
    np.ndarray of bool, shape target_shape
    """
    from scipy import ndimage

    h, w = target_shape
    projection = np.zeros((h, w), dtype=bool)

    if volume.sum() == 0:
        return projection

    X, Y, Z = grid.meshgrid()

    if view == "cc":
        col_proj, row_proj = geom.project_cc(X, Y, Z)
    else:
        col_proj, row_proj = geom.project_mlo(X, Y, Z)

    # Voxels inside the volume
    mask = volume > 0
    c_sel = col_proj[mask]
    r_sel = row_proj[mask]

    col = np.round(c_sel).astype(int)
    row = np.round(r_sel).astype(int)

    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    projection[row[valid], col[valid]] = True

    # Compute footprint based on voxel step in image space
    dx = (grid.x_max - grid.x_min) / max(grid.nx - 1, 1)
    dy = (grid.y_max - grid.y_min) / max(grid.ny - 1, 1)
    dz = (grid.z_max - grid.z_min) / max(grid.nz - 1, 1)

    r_col = max(int(np.ceil(dy / 2.0)), 1)
    if view == "cc":
        r_row = max(int(np.ceil(dx / 2.0)), 1)
    else:
        dr_mlo = dz * geom.cos_theta + dx * geom.sin_theta
        r_row = max(int(np.ceil(dr_mlo / 2.0)), 1)

    footprint = np.ones((2 * r_row + 1, 2 * r_col + 1), dtype=bool)
    dilated = ndimage.binary_dilation(projection, structure=footprint)
    filled = ndimage.binary_fill_holes(dilated)

    return filled


# ── Public entry point ─────────────────────────────────────────────────

def validate(
    cfg: Config,
    prep: PreprocessedData,
    geom: ProjectionGeometry,
    grid: VoxelGrid,
    breast_hull: np.ndarray,
    lesion_hull: np.ndarray,
    breast_mesh,
    lesion_mesh,
    lesion_centroid_tri: Tuple[float, float, float],
) -> ValidationMetrics:
    """Run all validation checks and return a metrics object."""

    log.info("Running validation …")
    metrics = ValidationMetrics()

    # ── Breast forward projections ─────────────────────────────────────
    proj_breast_cc = _forward_project_volume(
        breast_hull, grid, geom, "cc", prep.breast_cc.shape
    )
    proj_breast_mlo = _forward_project_volume(
        breast_hull, grid, geom, "mlo", prep.breast_mlo.shape
    )

    metrics.breast_dice_cc = dice_coefficient(proj_breast_cc, prep.breast_cc)
    metrics.breast_iou_cc = iou(proj_breast_cc, prep.breast_cc)
    metrics.breast_dice_mlo = dice_coefficient(proj_breast_mlo, prep.breast_mlo)
    metrics.breast_iou_mlo = iou(proj_breast_mlo, prep.breast_mlo)

    log.info("  Breast CC  Dice=%.4f  IoU=%.4f",
             metrics.breast_dice_cc, metrics.breast_iou_cc)
    log.info("  Breast MLO Dice=%.4f  IoU=%.4f",
             metrics.breast_dice_mlo, metrics.breast_iou_mlo)

    # ── Lesion forward projections ─────────────────────────────────────
    proj_lesion_cc = _forward_project_volume(
        lesion_hull, grid, geom, "cc", prep.roi_cc.shape
    )
    proj_lesion_mlo = _forward_project_volume(
        lesion_hull, grid, geom, "mlo", prep.roi_mlo.shape
    )

    metrics.lesion_dice_cc = dice_coefficient(proj_lesion_cc, prep.roi_cc)
    metrics.lesion_iou_cc = iou(proj_lesion_cc, prep.roi_cc)
    metrics.lesion_dice_mlo = dice_coefficient(proj_lesion_mlo, prep.roi_mlo)
    metrics.lesion_iou_mlo = iou(proj_lesion_mlo, prep.roi_mlo)

    log.info("  Lesion CC  Dice=%.4f  IoU=%.4f",
             metrics.lesion_dice_cc, metrics.lesion_iou_cc)
    log.info("  Lesion MLO Dice=%.4f  IoU=%.4f",
             metrics.lesion_dice_mlo, metrics.lesion_iou_mlo)

    # ── Mesh quality ───────────────────────────────────────────────────
    metrics.breast_vertices = len(breast_mesh.vertices)
    metrics.breast_faces = len(breast_mesh.faces)
    metrics.breast_is_watertight = bool(breast_mesh.is_watertight)
    metrics.breast_volume_voxels = int(breast_hull.sum())

    metrics.lesion_vertices = len(lesion_mesh.vertices)
    metrics.lesion_faces = len(lesion_mesh.faces)
    metrics.lesion_is_watertight = bool(lesion_mesh.is_watertight)
    metrics.lesion_volume_voxels = int(lesion_hull.sum())

    # ── Lesion centroid comparison ──────────────────────────────────────
    X, Y, Z = grid.meshgrid()
    if lesion_hull.sum() > 0:
        voxel_centroid = (
            float(X[lesion_hull].mean()),
            float(Y[lesion_hull].mean()),
            float(Z[lesion_hull].mean()),
        )
    else:
        voxel_centroid = (0.0, 0.0, 0.0)

    metrics.centroid_triangulated = lesion_centroid_tri
    metrics.centroid_voxel = voxel_centroid
    metrics.centroid_distance = float(np.linalg.norm(
        np.array(lesion_centroid_tri) - np.array(voxel_centroid)
    ))

    # ── Warnings ───────────────────────────────────────────────────────
    if metrics.breast_dice_cc < 0.7:
        metrics.notes.append(
            "Low breast Dice in CC view — the 3-D model may not "
            "accurately represent the breast in this projection."
        )
    if metrics.breast_dice_mlo < 0.7:
        metrics.notes.append(
            "Low breast Dice in MLO view — check the MLO angle parameter."
        )
    if metrics.lesion_dice_cc < 0.5:
        metrics.notes.append(
            "Low lesion Dice in CC view — the lesion shape may be poorly "
            "reconstructed from these masks."
        )
    if metrics.lesion_dice_mlo < 0.5:
        metrics.notes.append(
            "Low lesion Dice in MLO view."
        )
    if not metrics.breast_is_watertight:
        metrics.notes.append(
            "Breast mesh is not watertight — may have holes.  "
            "Try increasing voxel_resolution or smoothing."
        )
    metrics.notes.append(
        "With only 2 projection views, the 3-D reconstruction represents "
        "the visual hull — the maximal volume consistent with both "
        "silhouettes.  Intermediate cross-sections are NOT uniquely "
        "determined by the data."
    )

    # Store forward projections for visualisation
    metrics._proj_breast_cc = proj_breast_cc    # type: ignore[attr-defined]
    metrics._proj_breast_mlo = proj_breast_mlo  # type: ignore[attr-defined]
    metrics._proj_lesion_cc = proj_lesion_cc     # type: ignore[attr-defined]
    metrics._proj_lesion_mlo = proj_lesion_mlo   # type: ignore[attr-defined]

    log.info("Validation complete.")
    return metrics
