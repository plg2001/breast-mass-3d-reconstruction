"""
Modulo di Esportazione 3D (GLB / OBJ) con Materiali PBR
Esporta esclusivamente la superficie cutanea 3D pura della mammella.
Garantisce la compatibilità con Blender, Unity, 3ds Max e visualizzatori WebGL/glTF 2.0.
"""

import os
import json
import trimesh
from typing import Dict, Any


def export_reconstruction_to_glb(
    skin_mesh: trimesh.Trimesh,
    output_dir: str,
    patient_id: str,
    metrics: Dict[str, Any]
) -> Dict[str, str]:
    """
    Esporta la mesh 3D della sola superficie del seno in formato binario glTF (.glb)
    e Wavefront (.obj), con materiale PBR realistico opaco per la cute.
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}

    # Configurazione materiale PBR realistico e opaco per la superficie cutanea
    skin_pbr = trimesh.visual.material.PBRMaterial(
        name="Breast_Skin_Material",
        baseColorFactor=[0.86, 0.72, 0.64, 1.0],  # Tono cutaneo naturale e caldo (opaco)
        metallicFactor=0.0,
        roughnessFactor=0.45,
        doubleSided=True
    )
    skin_mesh.visual.material = skin_pbr

    # 1. Esporta il file GLB (glTF 2.0 binario)
    skin_glb_path = os.path.join(output_dir, f"{patient_id}_breast_surface.glb")
    with open(skin_glb_path, "wb") as f:
        f.write(skin_mesh.export(file_type="glb"))
    exported_files["skin_glb"] = skin_glb_path

    # 2. Esporta il file OBJ (mesh poligonale standard)
    skin_obj_path = os.path.join(output_dir, f"{patient_id}_breast_surface.obj")
    skin_mesh.export(skin_obj_path, file_type="obj")
    exported_files["skin_obj"] = skin_obj_path

    # 3. Salva il report metrico anatomico in formato JSON
    metrics_path = os.path.join(output_dir, f"{patient_id}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    exported_files["metrics_json"] = metrics_path

    return exported_files
