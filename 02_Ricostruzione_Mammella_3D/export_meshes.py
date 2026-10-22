"""
Mesh export module.

Exports the breast, internal tissues, and lesion meshes in GLB, OBJ, and STL formats.
Also creates a combined hierarchical scene in GLB for direct import into Unity or Blender.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .config import Config
from .internal_tissues import InternalTissues
from .utils import log


def _apply_colour(mesh: trimesh.Trimesh, rgba: tuple) -> trimesh.Trimesh:
    """Apply a uniform vertex colour to a mesh."""
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)),
    )
    return mesh


def export_mesh(
    mesh: trimesh.Trimesh,
    base_path: Path,
    formats: list,
    name: str,
) -> None:
    """Export a single mesh in the requested formats."""
    for fmt in formats:
        fmt = fmt.lower().strip(".")
        out = base_path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            mesh.export(str(out), file_type=fmt)
            log.info("  Exported %s → %s", name, out.name)
        except Exception as exc:
            log.warning("  Failed to export %s as %s: %s", name, fmt, exc)


def export_combined_scene(
    breast_mesh: trimesh.Trimesh,
    lesion_mesh: trimesh.Trimesh,
    out_path: Path,
    internal_tissues: Optional[InternalTissues] = None,
) -> None:
    """Create a combined multi-tissue GLB scene with separate nodes for Unity/Blender."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()

    # 1. Pectoral muscle (deep posterior base)
    if internal_tissues is not None:
        pec = internal_tissues.pectoral_mesh.copy()
        _apply_colour(pec, (175, 45, 45, 255))
        scene.add_geometry(pec, node_name="Pectoral_Muscle")

    # 2. Fibroglandular parenchyma (functional central core)
    if internal_tissues is not None:
        gland = internal_tissues.glandular_mesh.copy()
        _apply_colour(gland, (245, 175, 175, 220))
        scene.add_geometry(gland, node_name="Glandular_Tissue")

    # 3. Lactiferous ducts (branching ducts to nipple)
    if internal_tissues is not None:
        ducts = internal_tissues.ducts_mesh.copy()
        _apply_colour(ducts, (255, 205, 75, 255))
        scene.add_geometry(ducts, node_name="Lactiferous_Ducts")

    # 4. Lesion (pathological mass in UOQ)
    lesion_coloured = lesion_mesh.copy()
    _apply_colour(lesion_coloured, (220, 20, 20, 255))
    scene.add_geometry(lesion_coloured, node_name="Lesion")

    # 5. Outer breast envelope / skin & subcutaneous adipose layer (semi-transparent)
    breast_coloured = breast_mesh.copy()
    _apply_colour(breast_coloured, (235, 190, 170, 75))  # translucent skin
    scene.add_geometry(breast_coloured, node_name="Breast_Skin")

    try:
        scene.export(str(out_path), file_type="glb")
        log.info("  Combined scene → %s (Nodes: %s)", out_path.name, ", ".join(scene.geometry.keys()))
    except Exception as exc:
        log.warning("  Failed to export combined scene: %s", exc)


def export_all(
    cfg: Config,
    breast_mesh: trimesh.Trimesh,
    lesion_mesh: trimesh.Trimesh,
    internal_tissues: Optional[InternalTissues] = None,
) -> None:
    """Export all meshes in all configured formats."""
    log.info("Exporting meshes …")

    mesh_dir = cfg.output_path("meshes", "_placeholder").parent

    # Outer breast skin
    export_mesh(
        breast_mesh,
        mesh_dir / "breast",
        cfg.export_formats,
        "Breast_Skin",
    )

    # Internal tissues if present
    if internal_tissues is not None:
        export_mesh(
            internal_tissues.glandular_mesh,
            mesh_dir / "glandular",
            cfg.export_formats,
            "Glandular_Tissue",
        )
        export_mesh(
            internal_tissues.pectoral_mesh,
            mesh_dir / "pectoral",
            cfg.export_formats,
            "Pectoral_Muscle",
        )
        export_mesh(
            internal_tissues.ducts_mesh,
            mesh_dir / "ducts",
            cfg.export_formats,
            "Lactiferous_Ducts",
        )

    # Lesion
    export_mesh(
        lesion_mesh,
        mesh_dir / "lesion",
        cfg.export_formats,
        "Lesion",
    )

    # Combined scene (always GLB)
    export_combined_scene(
        breast_mesh,
        lesion_mesh,
        mesh_dir / "combined_scene.glb",
        internal_tissues=internal_tissues,
    )

    log.info("All exports saved to %s/meshes/", cfg.output_dir)
