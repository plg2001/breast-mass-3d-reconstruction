"""
Projection geometry for mammographic 3D reconstruction.

Coordinate system
─────────────────
We define a right-handed 3-D world frame aligned with clinical mammography:

    X : medial  <->  lateral  (row axis in CC projection)
    Y : posterior -> anterior (chest wall -> nipple, column/depth axis in both CC and MLO)
    Z : inferior  -> superior (longitudinal axis)

In standard mammography (e.g. CBIS-DDSM):
  • The chest wall is along the vertical edge of the image (column = 0).
  • Moving rightwards (columns) moves from chest wall toward nipple (Depth = Y).
  • In CC (cranio-caudal):
      The X-ray beam travels along Z (superior -> inferior).
      Detector is the XY plane:
          col_cc = y  (depth: chest wall -> nipple)
          row_cc = x  (medial-lateral position)

  • In MLO (medio-lateral oblique):
      The detector is tilted by angle theta (typically 45-60 deg) along the
      pectoral muscle.
      col_mlo = y                            (depth, along Posterior-Nipple Line)
      row_mlo = z * cos(theta) + x * sin(theta)  (oblique axis along pectoral line)

Triangulation / Back-projection
───────────────────────────────
Given corresponding points in CC and MLO:
    x = row_cc
    y = 0.5 * (col_cc + col_mlo)
    z = (row_mlo - x * sin(theta)) / cos(theta)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ProjectionGeometry:
    """Encapsulates CC and MLO projection parameters."""

    mlo_angle_deg: float = 45.0

    def __post_init__(self) -> None:
        self.mlo_angle_rad: float = float(np.radians(self.mlo_angle_deg))
        self.cos_theta: float = float(np.cos(self.mlo_angle_rad))
        self.sin_theta: float = float(np.sin(self.mlo_angle_rad))

    # ── forward projections ────────────────────────────────────────────

    def project_cc(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3-D points to the CC image plane.

        Returns (col_cc, row_cc).
            col_cc : depth (chest wall -> nipple) = y
            row_cc : medial-lateral = x
        """
        col_cc = y.copy()
        row_cc = x.copy()
        return col_cc, row_cc

    def project_mlo(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3-D points to the MLO image plane.

        Returns (col_mlo, row_mlo).
            col_mlo : depth = y
            row_mlo = z * cos(theta) + x * sin(theta)
        """
        col_mlo = y.copy()
        row_mlo = z * self.cos_theta + x * self.sin_theta
        return col_mlo, row_mlo

    # ── back-projection / triangulation ────────────────────────────────

    def triangulate(
        self,
        col_cc: float,
        row_cc: float,
        col_mlo: float,
        row_mlo: float,
    ) -> Tuple[float, float, float]:
        """Reconstruct the 3-D position of a point seen in both views.

        Parameters
        ----------
        col_cc, row_cc : float
            Position in the CC image (col=depth, row=ML).
        col_mlo, row_mlo : float
            Position in the MLO image (col=depth, row=oblique).

        Returns
        -------
        (x, y, z) in world coordinates.
        """
        x = float(row_cc)
        y = float(0.5 * (col_cc + col_mlo))
        z = float((row_mlo - x * self.sin_theta) / self.cos_theta)
        return x, y, z


@dataclass
class VoxelGrid:
    """Axis-aligned 3-D voxel grid with convenience accessors."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    resolution: int  # voxels along the longest axis

    def __post_init__(self) -> None:
        extent = np.array(
            [
                self.x_max - self.x_min,
                self.y_max - self.y_min,
                self.z_max - self.z_min,
            ]
        )
        longest = float(extent.max())
        if longest <= 0:
            raise ValueError("Voxel grid has zero or negative extent.")
        self.voxel_size: float = longest / self.resolution
        self.nx: int = max(int(round(extent[0] / self.voxel_size)), 1)
        self.ny: int = max(int(round(extent[1] / self.voxel_size)), 1)
        self.nz: int = max(int(round(extent[2] / self.voxel_size)), 1)
        self.shape: Tuple[int, int, int] = (self.nx, self.ny, self.nz)

    @property
    def total_voxels(self) -> int:
        return self.nx * self.ny * self.nz

    def linspace_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 1-D arrays for voxel-centre coordinates."""
        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        z = np.linspace(self.z_min, self.z_max, self.nz)
        return x, y, z

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 3-D coordinate arrays (indexing='ij')."""
        x, y, z = self.linspace_axes()
        return np.meshgrid(x, y, z, indexing="ij")

    def world_to_voxel(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert world coordinates to fractional voxel indices."""
        vi = (x - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1)
        vj = (y - self.y_min) / (self.y_max - self.y_min) * (self.ny - 1)
        vk = (z - self.z_min) / (self.z_max - self.z_min) * (self.nz - 1)
        return vi, vj, vk

    def summary(self) -> str:
        mem_mb = self.total_voxels * 1 / 1e6
        return (
            f"VoxelGrid {self.nx}x{self.ny}x{self.nz} = "
            f"{self.total_voxels:,} voxels "
            f"(~{mem_mb:.1f} MB bool) "
            f"voxel_size={self.voxel_size:.2f} px"
        )


def compute_grid_bounds(
    breast_bbox_cc: Tuple[int, int, int, int],
    breast_bbox_mlo: Tuple[int, int, int, int],
    geom: ProjectionGeometry,
    padding_frac: float = 0.05,
) -> Tuple[float, float, float, float, float, float]:
    """Compute the 3-D bounding box from breast bounding boxes in CC and MLO.

    Parameters
    ----------
    breast_bbox_cc : (col_min, row_min, col_max, row_max)
    breast_bbox_mlo : (col_min, row_min, col_max, row_max)
    geom : ProjectionGeometry
    padding_frac : float

    Returns
    -------
    (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    c_min_cc, r_min_cc, c_max_cc, r_max_cc = breast_bbox_cc
    c_min_mlo, r_min_mlo, c_max_mlo, r_max_mlo = breast_bbox_mlo

    # X range from CC rows (Medial-Lateral)
    x_min = float(r_min_cc)
    x_max = float(r_max_cc)

    # Y range from CC & MLO columns (Depth: chest wall -> nipple)
    y_min = float(min(c_min_cc, c_min_mlo))
    y_max = float(max(c_max_cc, c_max_mlo))

    # Z range from MLO geometry:
    # row_mlo = z * cos_theta + x * sin_theta
    # => z = (row_mlo - x * sin_theta) / cos_theta
    cos_t = geom.cos_theta
    sin_t = geom.sin_theta

    z_candidates = []
    for x in [x_min, x_max]:
        for r in [float(r_min_mlo), float(r_max_mlo)]:
            z_candidates.append((r - x * sin_t) / cos_t)

    z_min = min(z_candidates)
    z_max = max(z_candidates)

    # Add margin
    def pad(lo: float, hi: float) -> Tuple[float, float]:
        extent = hi - lo
        margin = max(extent * padding_frac, 2.0)
        return lo - margin, hi + margin

    x_min, x_max = pad(x_min, x_max)
    y_min, y_max = pad(y_min, y_max)
    z_min, z_max = pad(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max
