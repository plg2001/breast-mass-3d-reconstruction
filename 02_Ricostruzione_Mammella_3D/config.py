"""
Configuration module for the 3D mammography reconstruction pipeline.

All configurable parameters are collected here with sensible defaults.
Parameters can be overridden via command-line arguments or by editing
this file directly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class Config:
    """Master configuration for the reconstruction pipeline."""

    # ── Input paths ─────────────────────────────────────────────────────
    r_cc_path: str = ""          # Path to R-CC mammogram
    r_mlo_path: str = ""         # Path to R-MLO mammogram
    r_roi_cc_path: str = ""      # Path to R-ROI-CC lesion mask
    r_roi_mlo_path: str = ""     # Path to R-ROI-MLO lesion mask

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "output"   # Directory for all outputs

    # ── Reconstruction geometry ─────────────────────────────────────────
    # Reconstruction mode:
    # "anatomical" (default) -> smooth, organic, patient-specific anatomical dome
    #                           with cross-sections directly measured from CC and MLO
    # "visual_hull"          -> raw geometric ray intersection prism
    reconstruction_mode: str = "anatomical"

    # MLO angle in degrees measured from the horizontal plane.
    # Standard clinical MLO angle is 45–60°.  45° is the default.
    mlo_angle_deg: float = 45.0

    # Voxel grid resolution: number of voxels along the longest axis.
    # Higher values produce finer detail but require more memory/time.
    # 200 is a good balance (~32 MB per 3-D array).
    voxel_resolution: int = 200

    # Working image size: images are downsampled so that the longest
    # edge equals this value.  Keeps computation tractable while
    # preserving enough spatial detail for the visual hull.
    working_size: int = 512

    # ── Breast segmentation ────────────────────────────────────────────
    # Gaussian blur sigma (in pixels) applied before thresholding.
    breast_blur_sigma: float = 5.0

    # Minimum fraction of image area that the breast mask should occupy.
    # Used as a sanity check after segmentation.
    breast_min_area_frac: float = 0.05

    # ── Volume smoothing ───────────────────────────────────────────────
    # Gaussian sigma applied to the binary visual-hull volume before
    # isosurface extraction.  Removes voxel staircase artefacts.
    breast_volume_sigma: float = 2.0
    lesion_volume_sigma: float = 1.5

    # Marching-cubes isosurface threshold (after Gaussian blur of
    # binary volume, values are in [0, 1]).
    isosurface_level: float = 0.5

    # ── Mesh post-processing ───────────────────────────────────────────
    # Laplacian smoothing iterations and weight for the breast mesh.
    breast_smooth_iterations: int = 30
    breast_smooth_lambda: float = 0.5

    # Laplacian smoothing iterations and weight for the lesion mesh.
    lesion_smooth_iterations: int = 20
    lesion_smooth_lambda: float = 0.5

    # Maximum number of faces in the exported meshes (decimation).
    # Set to 0 to disable decimation.
    max_faces_breast: int = 50000
    max_faces_lesion: int = 20000

    # ── Export ──────────────────────────────────────────────────────────
    export_formats: List[str] = field(default_factory=lambda: ["glb", "obj"])

    # ── Validation ─────────────────────────────────────────────────────
    # Whether to generate forward-projection validation images.
    run_validation: bool = True

    # Whether to generate 3-D visualisation renders.
    run_3d_visualisation: bool = True

    # ── Pixel spacing (optional) ───────────────────────────────────────
    # If known, pixel spacing in mm/pixel for each view.  If None, the
    # pipeline works in pixel-coordinate units and notes the assumption.
    pixel_spacing_cc_mm: Optional[float] = None
    pixel_spacing_mlo_mm: Optional[float] = None

    # ── Derived paths (populated at runtime) ───────────────────────────
    def output_path(self, *parts: str) -> Path:
        """Return an output sub-path, creating directories as needed."""
        p = Path(self.output_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def validate_paths(self) -> None:
        """Check that all four input files exist."""
        for attr in ("r_cc_path", "r_mlo_path", "r_roi_cc_path", "r_roi_mlo_path"):
            p = Path(getattr(self, attr))
            if not p.is_file():
                raise FileNotFoundError(
                    f"Input file not found: {p}  (parameter: {attr})"
                )

    def save(self, path: Optional[str] = None) -> None:
        """Persist the current configuration to a JSON file."""
        if path is None:
            path = str(self.output_path("config.json"))
        data = {k: v for k, v in self.__dict__.items() if not callable(v)}
        # Convert non-serialisable types
        for k, v in data.items():
            if isinstance(v, Path):
                data[k] = str(v)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load a configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
