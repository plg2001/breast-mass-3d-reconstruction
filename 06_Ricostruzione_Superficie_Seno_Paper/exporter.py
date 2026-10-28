"""
Modulo di Esportazione 3D (GLB / OBJ) con Materiali PBR
Garantisce la compatibilità con Blender, Unity, 3ds Max e visualizzatori WebGL/glTF 2.0.
"""

import os
import json
import trimesh
from typing import Optional, Dict, Any


def export_reconstruction_to_glb(
    skin_mesh: trimesh.Trimesh,
    tumor_mesh: Optional[trimesh.Trimesh],
    output_dir: str,
    patient_id: str,
    metrics: Dict[str, Any]
) -> Dict[str, str]:
    """
    Esporta i modelli 3D in formato binario glTF (.glb) e Wavefront (.obj).
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}

    # 1. Configura il materiale PBR della pelle
    skin_pbr = trimesh.visual.material.PBRMaterial(
        name="Breast_Skin_Material",
        baseColorFactor=[0.85, 0.70, 0.62, 1.0],  # Tono organico cutaneo caldo
        metallicFactor=0.0,
        roughnessFactor=0.45,
        doubleSided=True
    )
    skin_mesh.visual.material = skin_pbr

    # 2. Esporta la superficie cutanea standalone in GLB e OBJ
    skin_glb_path = os.path.join(output_dir, f"{patient_id}_breast_surface.glb")
    skin_obj_path = os.path.join(output_dir, f"{patient_id}_breast_surface.obj")

    with open(skin_glb_path, "wb") as f:
        f.write(skin_mesh.export(file_type="glb"))
    exported_files["skin_glb"] = skin_glb_path

    skin_mesh.export(skin_obj_path, file_type="obj")
    exported_files["skin_obj"] = skin_obj_path

    # 3. Se è presente la lesione, crea una scena composita con pelle semitrasparente e tumore
    if tumor_mesh is not None:
        tumor_pbr = trimesh.visual.material.PBRMaterial(
            name="Tumor_Lesion_Material",
            baseColorFactor=[0.88, 0.12, 0.18, 1.0],  # Rosso rubino vivace
            metallicFactor=0.1,
            roughnessFactor=0.30,
            doubleSided=True
        )
        tumor_mesh.visual.material = tumor_pbr

        # Crea mesh pelle semitrasparente per visualizzare il tumore interno
        skin_translucent = skin_mesh.copy()
        skin_translucent_pbr = trimesh.visual.material.PBRMaterial(
            name="Breast_Skin_Translucent",
            baseColorFactor=[0.88, 0.74, 0.66, 0.55],  # Semitrasparente
            alphaMode="BLEND",
            metallicFactor=0.0,
            roughnessFactor=0.35,
            doubleSided=True
        )
        skin_translucent.visual.material = skin_translucent_pbr

        scene = trimesh.Scene()
        scene.add_geometry(skin_translucent, node_name="Breast_Skin_Envelope")
        scene.add_geometry(tumor_mesh, node_name="Tumor_Lesion_3D")

        composite_glb_path = os.path.join(output_dir, f"{patient_id}_breast_with_tumor.glb")
        with open(composite_glb_path, "wb") as f:
            f.write(scene.export(file_type="glb"))
        exported_files["composite_glb"] = composite_glb_path

    # 4. Salva il report metrico in formato JSON
    metrics_path = os.path.join(output_dir, f"{patient_id}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    exported_files["metrics_json"] = metrics_path

    return exported_files
