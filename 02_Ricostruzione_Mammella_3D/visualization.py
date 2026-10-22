"""
Diagnostic visualisations.

Generates publication-quality figures for:
  • Original mammograms with ROI overlays
  • Breast silhouette comparison (original vs. reconstructed)
  • Lesion ROI comparison
  • 3-D mesh renders from multiple viewpoints
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – used implicitly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh

from .config import Config
from .preprocessing import PreprocessedData
from .validation import ValidationMetrics
from .utils import log


# ── Colour maps ─────────────────────────────────────────────────────────

_OVERLAY_CMAP = ListedColormap(["none", "#ff000080"])  # transparent + red
_PROJ_CMAP = ListedColormap(["none", "#00ff0080"])  # transparent + green


# ── Helper: save figure ────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    log.info("  Saved: %s", path.name)


# ── 1. Input overview ──────────────────────────────────────────────────

def plot_input_overview(
    cfg: Config,
    prep: PreprocessedData,
) -> None:
    """2×2 grid: CC, MLO, ROI-on-CC, ROI-on-MLO."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), facecolor="black")
    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")

    # CC
    axes[0, 0].imshow(prep.cc, cmap="gray")
    axes[0, 0].set_title("R-CC", color="white", fontsize=14)

    # MLO
    axes[0, 1].imshow(prep.mlo, cmap="gray")
    axes[0, 1].set_title("R-MLO", color="white", fontsize=14)

    # CC + ROI overlay
    axes[1, 0].imshow(prep.cc, cmap="gray")
    axes[1, 0].imshow(prep.roi_cc, cmap=_OVERLAY_CMAP, vmin=0, vmax=1, alpha=0.5)
    axes[1, 0].set_title("R-CC + ROI", color="white", fontsize=14)

    # MLO + ROI overlay
    axes[1, 1].imshow(prep.mlo, cmap="gray")
    axes[1, 1].imshow(prep.roi_mlo, cmap=_OVERLAY_CMAP, vmin=0, vmax=1, alpha=0.5)
    axes[1, 1].set_title("R-MLO + ROI", color="white", fontsize=14)

    fig.suptitle("Input Images", color="white", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, cfg.output_path("figures", "01_input_overview.png"))


# ── 2. Breast silhouette comparison ────────────────────────────────────

def plot_breast_silhouette_comparison(
    cfg: Config,
    prep: PreprocessedData,
    metrics: ValidationMetrics,
) -> None:
    """Side-by-side: original breast mask vs. forward-projected mask."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="black")
    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")

    proj_cc = getattr(metrics, "_proj_breast_cc", None)
    proj_mlo = getattr(metrics, "_proj_breast_mlo", None)

    # CC row
    axes[0, 0].imshow(prep.breast_cc, cmap="gray")
    axes[0, 0].set_title("Original CC silhouette", color="white")

    if proj_cc is not None:
        axes[0, 1].imshow(proj_cc.astype(float), cmap="gray")
        axes[0, 1].set_title("Reconstructed CC projection", color="white")

        # Overlap
        overlap = np.zeros((*prep.breast_cc.shape, 3))
        overlap[..., 0] = prep.breast_cc        # red = original
        overlap[..., 1] = proj_cc.astype(float)  # green = reconstructed
        axes[0, 2].imshow(overlap)
        axes[0, 2].set_title(
            f"Overlap (Dice={metrics.breast_dice_cc:.3f})",
            color="white",
        )

    # MLO row
    axes[1, 0].imshow(prep.breast_mlo, cmap="gray")
    axes[1, 0].set_title("Original MLO silhouette", color="white")

    if proj_mlo is not None:
        axes[1, 1].imshow(proj_mlo.astype(float), cmap="gray")
        axes[1, 1].set_title("Reconstructed MLO projection", color="white")

        overlap = np.zeros((*prep.breast_mlo.shape, 3))
        overlap[..., 0] = prep.breast_mlo
        overlap[..., 1] = proj_mlo.astype(float)
        axes[1, 2].imshow(overlap)
        axes[1, 2].set_title(
            f"Overlap (Dice={metrics.breast_dice_mlo:.3f})",
            color="white",
        )

    fig.suptitle("Breast Silhouette Validation", color="white", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, cfg.output_path("figures", "02_breast_silhouette_validation.png"))


# ── 3. Lesion ROI comparison ──────────────────────────────────────────

def plot_lesion_roi_comparison(
    cfg: Config,
    prep: PreprocessedData,
    metrics: ValidationMetrics,
) -> None:
    """Similar to breast silhouette but for lesion ROI."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="black")
    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")

    proj_cc = getattr(metrics, "_proj_lesion_cc", None)
    proj_mlo = getattr(metrics, "_proj_lesion_mlo", None)

    # CC row
    axes[0, 0].imshow(prep.roi_cc, cmap="gray")
    axes[0, 0].set_title("Original ROI CC", color="white")

    if proj_cc is not None:
        axes[0, 1].imshow(proj_cc.astype(float), cmap="gray")
        axes[0, 1].set_title("Reconstructed lesion CC", color="white")

        overlap = np.zeros((*prep.roi_cc.shape, 3))
        overlap[..., 0] = prep.roi_cc
        overlap[..., 1] = proj_cc.astype(float)
        axes[0, 2].imshow(overlap)
        axes[0, 2].set_title(
            f"Overlap (Dice={metrics.lesion_dice_cc:.3f})",
            color="white",
        )

    # MLO row
    axes[1, 0].imshow(prep.roi_mlo, cmap="gray")
    axes[1, 0].set_title("Original ROI MLO", color="white")

    if proj_mlo is not None:
        axes[1, 1].imshow(proj_mlo.astype(float), cmap="gray")
        axes[1, 1].set_title("Reconstructed lesion MLO", color="white")

        overlap = np.zeros((*prep.roi_mlo.shape, 3))
        overlap[..., 0] = prep.roi_mlo
        overlap[..., 1] = proj_mlo.astype(float)
        axes[1, 2].imshow(overlap)
        axes[1, 2].set_title(
            f"Overlap (Dice={metrics.lesion_dice_mlo:.3f})",
            color="white",
        )

    fig.suptitle("Lesion ROI Validation", color="white", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, cfg.output_path("figures", "03_lesion_roi_validation.png"))


# ── 4. 3-D mesh renders ───────────────────────────────────────────────

def _mesh_to_poly3d(
    mesh: trimesh.Trimesh,
    color: str = "#5599bb",
    alpha: float = 0.35,
    edgecolor: str = "#337799",
    linewidth: float = 0.1,
) -> Poly3DCollection:
    """Convert a trimesh to a matplotlib Poly3DCollection."""
    verts = mesh.vertices
    tris = verts[mesh.faces]
    poly = Poly3DCollection(tris, alpha=alpha, linewidth=linewidth)
    poly.set_facecolor(color)
    poly.set_edgecolor(edgecolor)
    return poly


def _set_equal_aspect(ax, meshes):
    """Set equal aspect ratio for 3-D axes."""
    all_verts = np.vstack([m.vertices for m in meshes])
    mid = all_verts.mean(axis=0)
    max_range = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def plot_3d_mesh(
    cfg: Config,
    breast_mesh: trimesh.Trimesh,
    lesion_mesh: trimesh.Trimesh,
) -> None:
    """Render the 3-D meshes from four viewpoints."""
    viewpoints = [
        ("Front (CC view)", 0, -90),
        ("Side (ML view)", 0, 0),
        ("Top", 90, -90),
        ("Oblique (MLO-like)", 30, -45),
    ]

    fig = plt.figure(figsize=(20, 16), facecolor="black")
    fig.suptitle(
        "3-D Reconstruction — Breast + Lesion",
        color="white", fontsize=16, y=0.98,
    )

    for idx, (title, elev, azim) in enumerate(viewpoints, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d", facecolor="black")
        ax.view_init(elev=elev, azim=azim)

        # Breast (translucent)
        poly_breast = _mesh_to_poly3d(breast_mesh, color="skyblue", alpha=0.15)
        ax.add_collection3d(poly_breast)

        # Lesion (opaque red)
        poly_lesion = _mesh_to_poly3d(lesion_mesh, color="red", alpha=0.8)
        ax.add_collection3d(poly_lesion)

        _set_equal_aspect(ax, [breast_mesh, lesion_mesh])

        ax.set_xlabel("X (ML)", color="gray", fontsize=8)
        ax.set_ylabel("Y (AP)", color="gray", fontsize=8)
        ax.set_zlabel("Z (SI)", color="gray", fontsize=8)
        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.tick_params(colors="gray", labelsize=6)

        # Style the 3-D panes
        ax.xaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))
        ax.yaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))
        ax.zaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, cfg.output_path("figures", "04_3d_reconstruction.png"), dpi=200)


# ── 5. Combined validation summary ────────────────────────────────────

def plot_validation_summary(
    cfg: Config,
    metrics: ValidationMetrics,
) -> None:
    """Bar charts of Dice and IoU metrics."""
    labels = ["Breast CC", "Breast MLO", "Lesion CC", "Lesion MLO"]
    dice_vals = [
        metrics.breast_dice_cc,
        metrics.breast_dice_mlo,
        metrics.lesion_dice_cc,
        metrics.lesion_dice_mlo,
    ]
    iou_vals = [
        metrics.breast_iou_cc,
        metrics.breast_iou_mlo,
        metrics.lesion_iou_cc,
        metrics.lesion_iou_mlo,
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    bars1 = ax.bar(x - width / 2, dice_vals, width, label="Dice", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, iou_vals, width, label="IoU", color="#dd8452")

    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics — Forward Projection Match")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    _save(fig, cfg.output_path("figures", "05_validation_summary.png"))


# ── 6. Internal anatomical tissues 3-D view ───────────────────────────

def plot_internal_tissues(
    cfg: Config,
    breast_mesh: trimesh.Trimesh,
    lesion_mesh: trimesh.Trimesh,
    internal_tissues,
) -> None:
    """Render internal tissues in 3-D with transparency and dissection views."""
    if internal_tissues is None:
        return

    fig = plt.figure(figsize=(20, 16), facecolor="black")
    fig.suptitle(
        "Internal Anatomical Tissues — Glandular Core, Pectoral Muscle, Ducts & Lesion",
        color="white", fontsize=16, y=0.98,
    )

    views = [
        ("Transparent Full Anatomy (Skin + Gland + Ducts + Lesion)", 20, 65, True),
        ("Dissected Parenchyma Core & Lesion (No Skin)", 20, 65, False),
        ("Sagittal Side View (Anatomical Depth Profile)", 0, 0, True),
        ("Axial Top View (Transverse Section)", 90, -90, True),
    ]

    gland_mesh = internal_tissues.glandular_mesh
    pec_mesh = internal_tissues.pectoral_mesh
    ducts_mesh = internal_tissues.ducts_mesh

    for idx, (title, elev, azim, show_skin) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d", facecolor="black")
        ax.view_init(elev=elev, azim=azim)

        # 1. Pectoral muscle (deep red)
        poly_pec = _mesh_to_poly3d(pec_mesh, color="#a02828", alpha=0.85, edgecolor="#701818", linewidth=0.1)
        ax.add_collection3d(poly_pec)

        # 2. Fibroglandular parenchyma (soft pinkish-peach)
        poly_gland = _mesh_to_poly3d(gland_mesh, color="#f5a5a5", alpha=0.55, edgecolor="#d07070", linewidth=0.1)
        ax.add_collection3d(poly_gland)

        # 3. Lactiferous ducts (gold)
        poly_ducts = _mesh_to_poly3d(ducts_mesh, color="#ffd700", alpha=0.90, edgecolor="#cca000", linewidth=0.1)
        ax.add_collection3d(poly_ducts)

        # 4. Lesion (bright ruby red)
        poly_les = _mesh_to_poly3d(lesion_mesh, color="#ff1111", alpha=0.95, edgecolor="#990000", linewidth=0.2)
        ax.add_collection3d(poly_les)

        # 5. Outer skin envelope (optional / translucent)
        if show_skin:
            poly_skin = _mesh_to_poly3d(breast_mesh, color="#4488aa", alpha=0.10, edgecolor="#336688", linewidth=0.05)
            ax.add_collection3d(poly_skin)

        _set_equal_aspect(ax, [breast_mesh, pec_mesh])

        ax.set_xlabel("X (ML)", color="gray", fontsize=8)
        ax.set_ylabel("Y (AP)", color="gray", fontsize=8)
        ax.set_zlabel("Z (SI)", color="gray", fontsize=8)
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.tick_params(colors="gray", labelsize=6)

        ax.xaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))
        ax.yaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))
        ax.zaxis.pane.set_facecolor((0.1, 0.1, 0.1, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, cfg.output_path("figures", "06_internal_tissues_3d.png"), dpi=200)


# ── Public entry point ─────────────────────────────────────────────────

def generate_all_figures(
    cfg: Config,
    prep: PreprocessedData,
    breast_mesh: trimesh.Trimesh,
    lesion_mesh: trimesh.Trimesh,
    metrics: ValidationMetrics,
    internal_tissues=None,
) -> None:
    """Generate and save all diagnostic figures."""
    log.info("Generating visualisations …")
    plot_input_overview(cfg, prep)
    plot_breast_silhouette_comparison(cfg, prep, metrics)
    plot_lesion_roi_comparison(cfg, prep, metrics)
    if cfg.run_3d_visualisation:
        plot_3d_mesh(cfg, breast_mesh, lesion_mesh)
        if internal_tissues is not None:
            plot_internal_tissues(cfg, breast_mesh, lesion_mesh, internal_tissues)
    plot_validation_summary(cfg, metrics)
    log.info("All figures saved to %s/figures/", cfg.output_dir)
