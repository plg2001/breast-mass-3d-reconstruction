"""
Entry point principale per la Ricostruzione 3D della Superficie del Seno da Mammografie 2D
Basato sulla letteratura scientifica:
- Huang et al. (2006) - Bi-planar MPGS alignment & silhouette projection
- Pinto et al. (2023) - Generative compressed breast shape model
- Rodríguez-Ruiz et al. (2017) - In vivo shape characterization
"""

import os
import sys
import argparse
import subprocess
from typing import Optional, Tuple, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from segmentation import process_mammogram_view
from kinematics import BiplanarKinematics
from surface_reconstruction import BreastSurfaceReconstructor
from exporter import export_reconstruction_to_glb


def resolve_data_directory(data_dir: str) -> str:
    """
    Risolve il percorso assoluto della cartella data.
    """
    candidates = [
        data_dir,
        os.path.join(CURRENT_DIR, data_dir),
        os.path.join(PROJECT_ROOT, data_dir),
        os.path.join(CURRENT_DIR, "..", "data"),
        os.path.join(PROJECT_ROOT, "data"),
    ]
    for c in candidates:
        abs_c = os.path.abspath(c)
        if os.path.exists(abs_c) and os.path.isdir(abs_c):
            return abs_c
    raise FileNotFoundError(
        f"Cartella dati non trovata. Tentativi: {[os.path.abspath(c) for c in candidates]}"
    )


def resolve_patient_id(data_dir: str, patient_id_input: str) -> str:
    """
    Normalizza e risolve l'ID del paziente gestendo formati flessibili
    (es. '81', 'P_81', '00081', 'P_00081').
    """
    clean = patient_id_input.strip()
    direct_path = os.path.join(data_dir, clean)
    if os.path.exists(direct_path):
        return clean

    digits = ''.join(c for c in clean if c.isdigit())
    if digits:
        formatted_5 = f"P_{int(digits):05d}"
        if os.path.exists(os.path.join(data_dir, formatted_5)):
            return formatted_5

        for folder in os.listdir(data_dir):
            if folder.endswith(digits) or folder.lower() == clean.lower():
                return folder

    return clean


def find_patient_mammogram_files(
    patient_dir: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Rileva automaticamente i file radiografici CC ed MLO escludendo qualsiasi ROI.
    """
    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Cartella paziente non trovata: {patient_dir}")

    files = os.listdir(patient_dir)

    cc_file = None
    mlo_file = None

    for f in sorted(files):
        f_lower = f.lower()
        if f_lower.endswith(('.jpg', '.jpeg', '.png', '.dcm')):
            # Ignora esplicitamente file ROI o annotazioni di lesione
            if f.startswith('ROI-') or 'roi' in f_lower or 'mask' in f_lower:
                continue

            if 'cc' in f_lower and cc_file is None:
                cc_file = os.path.join(patient_dir, f)
            elif 'mlo' in f_lower and mlo_file is None:
                mlo_file = os.path.join(patient_dir, f)

    return cc_file, mlo_file


def run_blender_automation(
    blender_exe: str,
    glb_path: str,
    blend_path: str,
    preview_png_path: str
) -> bool:
    """
    Avvia Blender in modalità headless per configurare la scena 3D, luci, camera e salvare .blend e preview.
    """
    if not os.path.exists(blender_exe):
        return False

    script_path = os.path.join(CURRENT_DIR, "blender_viewer.py")
    cmd = [
        blender_exe,
        "--background",
        "--python", script_path,
        "--",
        os.path.abspath(glb_path),
        os.path.abspath(blend_path),
        os.path.abspath(preview_png_path)
    ]

    try:
        print("[INFO] Generazione scena Blender e rendering anteprima in corso...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"[AVVISO] Impossibile eseguire l'automazione Blender: {e}")
        return False


def run_pipeline(
    patient_id: str,
    data_dir: str = "data",
    output_base_dir: str = os.path.join(CURRENT_DIR, "output"),
    pixel_spacing_mm: float = 0.05,
    blender_exe: Optional[str] = None
) -> Dict[str, Any]:
    """
    Esegue l'intera pipeline di ricostruzione della sola superficie mammaria per il paziente specificato.
    """
    resolved_data_dir = resolve_data_directory(data_dir)
    resolved_patient_id = resolve_patient_id(resolved_data_dir, patient_id)

    print("=" * 70)
    print(f" RICOSTRUZIONE 3D SUPERFICIE MAMMARIA (Paper Huang/Pinto/Kita)")
    print(f" Paziente ID: {resolved_patient_id} (input: {patient_id})")
    print(f" Cartella Dati: {resolved_data_dir}")
    print("=" * 70)

    patient_dir = os.path.join(resolved_data_dir, resolved_patient_id)
    cc_path, mlo_path = find_patient_mammogram_files(patient_dir)

    if not cc_path or not mlo_path:
        raise FileNotFoundError(f"Impossibile trovare la coppia di viste CC ed MLO in {patient_dir}")

    print(f"[1/4] Vista CC rilevata:  {os.path.basename(cc_path)}")
    print(f"[1/4] Vista MLO rilevata: {os.path.basename(mlo_path)}")

    # 1. Segmentazione border ed estrazione contorni cutanei
    print("\n[2/4] Segmentazione anatomica bordo cutaneo e linea PNL...")
    cc_data = process_mammogram_view(cc_path)
    mlo_data = process_mammogram_view(mlo_path)

    # 2. Allineamento cinematico bi-planare
    print("\n[3/4] Allineamento cinematico bi-planare e normalizzazione metrica...")
    kinematics = BiplanarKinematics(
        cc_data=cc_data,
        mlo_data=mlo_data,
        pixel_spacing_mm=pixel_spacing_mm
    )

    # 3. Lofting super-ellittico continuo e generazione mesh stagna
    print("\n[4/4] Generazione mesh 3D superficie cutanea (super-ellissi p(t))...")
    reconstructor = BreastSurfaceReconstructor(kinematics)
    skin_mesh = reconstructor.reconstruct_surface_mesh()
    metrics = reconstructor.compute_anatomical_metrics(skin_mesh)

    # 4. Esportazione in GLB / OBJ della sola superficie cutanea
    patient_output_dir = os.path.join(output_base_dir, resolved_patient_id)
    print(f"\nEsportazione file 3D superficie in: {patient_output_dir}")
    exported_files = export_reconstruction_to_glb(
        skin_mesh=skin_mesh,
        output_dir=patient_output_dir,
        patient_id=resolved_patient_id,
        metrics=metrics
    )

    # 5. Automazione Blender
    chosen_glb = exported_files["skin_glb"]
    blend_path = os.path.join(patient_output_dir, f"{resolved_patient_id}_scene.blend")
    preview_png_path = os.path.join(patient_output_dir, f"{resolved_patient_id}_preview.png")

    if blender_exe and os.path.exists(blender_exe):
        blender_ok = run_blender_automation(blender_exe, chosen_glb, blend_path, preview_png_path)
        if blender_ok:
            exported_files["blend_file"] = blend_path
            exported_files["preview_png"] = preview_png_path

    height_val = metrics.get('height_vertical_mm', metrics.get('height_mm', 0.0))
    print("\n" + "=" * 70)
    print(" RICOSTRUZIONE SUPERFICIE 3D COMPLETATA CON SUCCESSO!")
    print("=" * 70)
    print(f"  * Volume mammella stimato: {metrics['volume_cm3']} cm³")
    print(f"  * Area superficiale cute:  {metrics['surface_area_cm2']} cm²")
    print(f"  * Dimensioni (L x P x A):   {metrics['width_mm']} x {metrics['depth_pnl_mm']} x {height_val} mm (Verticale: {height_val} mm)")
    print(f"  * Chiusura stagna:         {metrics['is_watertight']} (Watertight Manifold)")
    print(f"  * File GLB Superficie:     {exported_files['skin_glb']}")
    if "blend_file" in exported_files:
        print(f"  * File Scena Blender:      {exported_files['blend_file']}")
        print(f"  * Render Anteprima PNG:    {exported_files['preview_png']}")
    print("=" * 70)

    return exported_files


def main():
    default_blender = r"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"
    if not os.path.exists(default_blender):
        default_blender = "blender"

    parser = argparse.ArgumentParser(
        description="Ricostruzione 3D Superficie Seno da Mammografie 2D (Paper Huang / Pinto / Kita)"
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default="P_00081",
        help="ID del paziente (es. P_00081, 81, P_00001, P_00005)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Percorso alla cartella contenente i dati dei pazienti"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "output"),
        help="Cartella di destinazione per i file 3D e i report"
    )
    parser.add_argument(
        "--pixel_spacing_mm",
        type=float,
        default=0.05,
        help="Calibrazione metrica detector mammografico (mm/pixel)"
    )
    parser.add_argument(
        "--blender_exe",
        type=str,
        default=default_blender,
        help="Percorso dell'eseguibile di Blender"
    )

    args = parser.parse_args()

    run_pipeline(
        patient_id=args.patient_id,
        data_dir=args.data_dir,
        output_base_dir=args.output_dir,
        pixel_spacing_mm=args.pixel_spacing_mm,
        blender_exe=args.blender_exe
    )


if __name__ == "__main__":
    main()
