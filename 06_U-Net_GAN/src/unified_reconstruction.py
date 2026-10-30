"""
Pipeline Unificata: Ricostruzione Anatomica Superficie Seno (Modulo 06) + 3D U-Net-GAN Lesione (Modulo 07)

Combina:
1. La superficie cutanea continua, liscia e stagna della mammella (Huang et al. 2006 / Pinto et al. 2023)
2. Il tessuto parenchimale/fibroglandolare interno ricco di rugosita biologiche e dotti predetto da 3D U-Net-GAN
3. La localizzazione volumetrica 3D del tumore con la rete neurale 3D U-Net-GAN e annotazioni cliniche ROI

Risultato:
Un modello 3D medico completo (GLB + .blend) con seno anatomico realistico, parenchima rugoso interno
e tumore focalizzato all'interno.
"""

import os
import sys
import argparse
import numpy as np
import scipy.ndimage as ndimage
import trimesh
from skimage.measure import marching_cubes

# Percorsi ai moduli
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODULE_05_DIR = os.path.join(PROJECT_ROOT, "05_Ricostruzione_Superficie_Seno_Paper")
MODULE_06_DIR = os.path.dirname(CURRENT_DIR)

if MODULE_05_DIR not in sys.path:
    sys.path.insert(0, MODULE_05_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from segmentation import process_mammogram_view
from kinematics import BiplanarKinematics
from surface_reconstruction import BreastSurfaceReconstructor
from tumor_localization import TumorReconstructor
from inference import (
    load_generator, preprocess_image, run_inference, 
    resolve_data_directory, Config, safe_smooth_mesh
)
from victre_anatomy import (
    generate_victre_ductal_tree,
    generate_cooper_ligaments,
    generate_subcutaneous_fat,
    generate_pectoral_muscle,
    generate_superficial_veins,
    generate_deep_arteries,
    generate_nipple_areola_complex,
    generate_lymph_nodes,
    generate_retromammary_fat,
    generate_mammary_nerves,
    generate_lymphatic_vessels,
    generate_deep_pectoral_fascia
)


def sanitize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Assicura che la mesh sia geometricamente valida per esportazione glTF/GLB e Blender:
    - Rimuove vertici e facce degenerate o infinite
    - Elimina qualsiasi NaN/Inf nei vertici e nelle normali
    - Previene l'errore Blender 'Bad glTF: json contained NaN'
    """
    if mesh is None or len(mesh.vertices) == 0:
        return mesh
    
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()
    except Exception:
        pass

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if not np.all(np.isfinite(verts)):
        verts = np.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)
        mesh.vertices = verts

    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        if not np.all(np.isfinite(normals)) or len(normals) != len(mesh.vertices):
            mesh.recompute_normals()
            normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
            if not np.all(np.isfinite(normals)):
                normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normals = normals / norms
                mesh.vertex_normals = normals
    except Exception:
        pass

    return mesh


def find_patient_views(patient_dir: str):
    """Trova le viste CC e MLO (escludendo maschere ROI)."""
    files = [f for f in os.listdir(patient_dir) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')) and not f.upper().startswith("ROI-")]
    
    cc_file, mlo_file = None, None
    for f in files:
        if "CC" in f.upper() and not cc_file:
            cc_file = os.path.join(patient_dir, f)
        elif "MLO" in f.upper() and not mlo_file:
            mlo_file = os.path.join(patient_dir, f)
            
    return cc_file, mlo_file


def find_patient_rois(patient_dir: str):
    """Trova le maschere ROI di annotazione clinica della lesione (CC e MLO)."""
    files = [f for f in os.listdir(patient_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm'))]
    roi_cc, roi_mlo = None, None
    for f in files:
        f_upper = f.upper()
        if "ROI" in f_upper:
            if "CC" in f_upper and not roi_cc:
                roi_cc = os.path.join(patient_dir, f)
            elif "MLO" in f_upper and not roi_mlo:
                roi_mlo = os.path.join(patient_dir, f)
    return roi_cc, roi_mlo


def build_anatomical_rugose_parenchyma(kinematics, vol_3d, skin_mesh):
    """
    Costruisce la mesh 3D del parenchima/tessuto fibroglandolare interno campionando
    la densita rugosa, complessa e organica predetta dalla rete neurale 3D U-Net-GAN (Modulo 07)
    e mappandola in modo CONFORME all'involucro anatomico super-ellittico del seno (Modulo 06).
    
    Caratteristiche:
    - Conformita anatomica continua: il tessuto interno segue perfettamente la sagoma,
      la curvatura del polo superiore e la convergenza anteriore verso il capezzolo.
    - Margine adiposo sottocutaneo uniforme: rispetta uno spessore proporzionale costante (13% dei semiassi).
    - Preserva le convoluzioni, i lobi, i solchi e le rugosita biologiche del modello 3D U-Net-GAN.
    - Esente da NaN o facce degenerate (stabile per esportazione GLTF/GLB e Blender).
    """
    Nx, Ny, Nz = 80, 96, 80
    min_b, max_b = skin_mesh.bounds
    xs = np.linspace(min_b[0], max_b[0], Nx)
    ys = np.linspace(min_b[1], max_b[1], Ny)
    zs = np.linspace(0.0, kinematics.pnl_mm, Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    grid_density = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    for k, z_val in enumerate(zs):
        t_ratio = z_val / kinematics.pnl_mm
        slice_idx = int(np.clip(t_ratio * (kinematics.num_depth_slices - 1), 0, kinematics.num_depth_slices - 1))
        _, W, H, Xc, Zc = kinematics.get_slice_parameters(slice_idx)
        
        # Margine adiposo sottocutaneo anatomico: 87% dei semiassi della cute (13% spazio adiposo uniforme)
        a = max(1.5, 0.5 * W * 0.87)
        b = max(1.5, 0.5 * H * 0.87)
        p = 2.15 - 0.15 * t_ratio
        
        # Distanza radiale normalizzata all'interno della super-ellisse della fetta
        dx = np.abs(X[:, :, k] - Xc) / a
        dy = np.abs(Y[:, :, k] - Zc) / b
        dist_se = (dx ** p + dy ** p) ** (1.0 / p)
        
        # Involucro anatomico conforme: 1.0 all'interno, caduta controllata entro il margine
        base_envelope = np.clip((1.02 - dist_se) / 0.16, 0.0, 1.0)
        
        # Coordinate locali normalizzate per fetta per campionare la rete neurale [0, 1]
        u = np.clip(((X[:, :, k] - Xc) / a + 1.0) * 0.5, 0.0, 1.0)
        v = np.clip(((Y[:, :, k] - Zc) / b + 1.0) * 0.5, 0.0, 1.0)
        
        d_coord = u * 9.0
        h_coord = (1.0 - v) * 127.0
        w_coord = np.clip(t_ratio * 127.0, 0.0, 127.0)
        
        coords = np.stack([d_coord, h_coord, np.full_like(d_coord, w_coord)], axis=0)
        neural_sample = ndimage.map_coordinates(vol_3d, coords, order=2, mode='nearest')
        
        # Delimitazione anatomica toracica / muscolo pettorale:
        # Nella proiezione MLO, il muscolo pettorale occupa la regione supero-posteriore (alto v, basso t_ratio).
        # Il vero parenchima ghiandolare non invade il muscolo pettorale né l'apice del torace.
        if t_ratio < 0.45:
            v_pec = 0.62 + 0.35 * (t_ratio / 0.45)
            pec_falloff = np.clip((v_pec - v) / 0.12, 0.0, 1.0)
        else:
            pec_falloff = 1.0
            
        # Spazio adiposo retromammario alla base toracica (t_ratio -> 0)
        retromammary_falloff = np.clip(t_ratio / 0.04, 0.0, 1.0)
        
        # Convergenza anteriore verso il capezzolo (t_ratio -> 1)
        nipple_falloff = np.clip((0.99 - t_ratio) / 0.04, 0.0, 1.0) if t_ratio > 0.95 else 1.0
        
        # Modulazione della densita con la texture neurale biologica (valli, creste e lobuli)
        neural_mod = 0.20 + 0.80 * neural_sample
        combined = base_envelope * neural_mod * pec_falloff * retromammary_falloff * nipple_falloff
        grid_density[:, :, k] = combined

    spacing = (
        (max_b[0] - min_b[0]) / (Nx - 1),
        (max_b[1] - min_b[1]) / (Ny - 1),
        kinematics.pnl_mm / (Nz - 1)
    )
    smooth_grid = ndimage.gaussian_filter(grid_density, sigma=0.38)
    verts, faces, normals, _ = marching_cubes(smooth_grid, level=0.32, spacing=spacing)
    verts[:, 0] += min_b[0]
    verts[:, 1] += min_b[1]
    verts[:, 2] += 0.0

    tissue_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    safe_smooth_mesh(tissue_mesh, lamb=0.20, iterations=1)
    tissue_mesh = sanitize_mesh(tissue_mesh)
    
    # Colore ambrato/arancio tenue semitrasparente per il tessuto interno
    tissue_color = np.array([235, 170, 115, 120], dtype=np.uint8)
    tissue_mesh.visual.vertex_colors = np.tile(tissue_color, (len(tissue_mesh.vertices), 1))
    return tissue_mesh


def run_unified_pipeline(
    patient_id: str,
    data_dir: str = None,
    output_dir: str = None,
    blender_exe: str = None
):
    print("=" * 75)
    print("   RICOSTRUZIONE UNIFICATA: SUPERFICIE ANATOMICA 3D + U-NET-GAN LESIONE")
    print(f"   Paziente: {patient_id}")
    print("=" * 75)
    
    # Risoluzione cartella dati paziente
    resolved_data_dir = data_dir or resolve_data_directory()
    if not os.path.exists(resolved_data_dir):
        raise FileNotFoundError(f"Cartella dati non trovata: {resolved_data_dir}")
        
    pid_num = patient_id.replace("P_", "").replace("p_", "").lstrip("0") or "0"
    candidates = [
        patient_id,
        f"P_{int(pid_num):05d}",
        f"P_{int(pid_num)}",
        f"Calc-Test_P_{int(pid_num):05d}_LEFT_CC",
        f"Mass-Test_P_{int(pid_num):05d}_LEFT_CC",
        f"Calc-Training_P_{int(pid_num):05d}_LEFT_CC",
        f"Mass-Training_P_{int(pid_num):05d}_LEFT_CC",
    ]
    
    patient_dir = None
    resolved_pid = patient_id
    for c in candidates:
        cand_dir = os.path.join(resolved_data_dir, c)
        if os.path.exists(cand_dir) and os.path.isdir(cand_dir):
            patient_dir = cand_dir
            resolved_pid = c
            break
            
    if not patient_dir:
        raise FileNotFoundError(f"Paziente {patient_id} non trovato in {resolved_data_dir}")
        
    cc_path, mlo_path = find_patient_views(patient_dir)
    if not cc_path or not mlo_path:
        raise FileNotFoundError(f"Coppia di viste CC e MLO non trovata in {patient_dir}")
        
    print(f"\n[1/4] Viste caricate:")
    print(f"  * Vista CC : {os.path.basename(cc_path)}")
    print(f"  * Vista MLO: {os.path.basename(mlo_path)}")
    
    # -----------------------------------------------------------------
    # FASE 1: Ricostruzione Superficie Anatomica Esterna (Modulo 06)
    # -----------------------------------------------------------------
    print("\n[2/4] Generazione superficie cutanea anatomica liscia (Modulo 06)...")
    cc_data = process_mammogram_view(cc_path)
    mlo_data = process_mammogram_view(mlo_path)
    kinematics = BiplanarKinematics(cc_data, mlo_data, pixel_spacing_mm=0.05)
    reconstructor = BreastSurfaceReconstructor(kinematics)
    skin_mesh = reconstructor.reconstruct_surface_mesh()
    skin_mesh = sanitize_mesh(skin_mesh)
    metrics = reconstructor.compute_anatomical_metrics(skin_mesh)
    
    min_b, max_b = skin_mesh.bounds
    print(f"  * Cute generata: {len(skin_mesh.vertices)} vertici, {len(skin_mesh.faces)} facce")
    print(f"  * Dimensioni anatomiche: {metrics['width_mm']} x {metrics['depth_pnl_mm']} x {metrics.get('height_vertical_mm', 0)} mm")
    print(f"  * Volume: {metrics['volume_cm3']} cm³")
    
    # Colore rosa tenue molto trasparente per la cute
    skin_color = np.array([255, 205, 195, 60], dtype=np.uint8)
    skin_mesh.visual.vertex_colors = np.tile(skin_color, (len(skin_mesh.vertices), 1))
    
    # -----------------------------------------------------------------
    # FASE 2: Inferenza 3D U-Net-GAN + Anatomia Iper-Realistica FDA VICTRE
    # -----------------------------------------------------------------
    print("\n[3/4] Generazione strati anatomici iper-realistici (FDA VICTRE + 3D U-Net-GAN)...")
    weights_path = os.path.join(MODULE_06_DIR, "training result", "unet3d_generator.pth")
    generator = load_generator(weights_path)
    
    # Preprocessing clinico dell'immagine MLO
    img_2d = preprocess_image(mlo_path)
    volume_3d = run_inference(generator, img_2d)
    
    scene = trimesh.Scene()
    scene.add_geometry(skin_mesh, node_name="breast_skin")
    
    # 1. Complesso Areola-Capezzolo (NAC) conforme con papilla e ghiandole di Montgomery
    print("  * [1/10] Generazione complesso areola-capezzolo (NAC)...")
    nac_mesh = generate_nipple_areola_complex(kinematics, skin_mesh)
    nac_mesh = sanitize_mesh(nac_mesh)
    scene.add_geometry(nac_mesh, node_name="nipple_areola")

    # 2. Plesso venoso superficiale sottocutaneo (Cerchio di Haller + vene toraco-epigastriche)
    print("  * [2/10] Generazione plesso venoso superficiale (Circulus venosus Halleri)...")
    veins_mesh = generate_superficial_veins(kinematics, skin_mesh)
    veins_mesh = sanitize_mesh(veins_mesh)
    scene.add_geometry(veins_mesh, node_name="breast_veins")

    # 3. Rete arteriosa ghiandolare profonda (mammaria interna + toracica laterale)
    print("  * [3/10] Generazione rete arteriosa ghiandolare profonda (nutrita da densità U-Net-GAN)...")
    arteries_mesh = generate_deep_arteries(kinematics, volume_3d=volume_3d)
    arteries_mesh = sanitize_mesh(arteries_mesh)
    scene.add_geometry(arteries_mesh, node_name="breast_arteries")

    # 4. Mantello adiposo premammario sottocutaneo (VICTRE compartment.cxx)
    print("  * [4/10] Generazione mantello adiposo premammario sottocutaneo...")
    fat_mesh = generate_subcutaneous_fat(kinematics, skin_mesh)
    fat_mesh = sanitize_mesh(fat_mesh)
    scene.add_geometry(fat_mesh, node_name="subcutaneous_fat")

    # 5. Parenchima fibroglandolare rugoso campionato da 3D U-Net-GAN
    print("  * [5/10] Generazione parenchima fibroglandolare rugoso...")
    tissue_mesh = build_anatomical_rugose_parenchyma(kinematics, volume_3d, skin_mesh)
    scene.add_geometry(tissue_mesh, node_name="fibroglandular_tissue")
    print(f"    - Vertici: {len(tissue_mesh.vertices)}, Facce: {len(tissue_mesh.faces)}")

    # 6. Albero galattoforo duttale guidato da densità U-Net-GAN (VICTRE duct.cxx)
    print("  * [6/10] Generazione albero duttale galattoforo guidato da densità 3D U-Net-GAN...")
    duct_mesh = generate_victre_ductal_tree(kinematics, volume_3d=volume_3d, num_lobes=14)
    duct_mesh = sanitize_mesh(duct_mesh)
    scene.add_geometry(duct_mesh, node_name="lactiferous_ducts")
    print(f"    - Vertici: {len(duct_mesh.vertices)}, Facce: {len(duct_mesh.faces)}")

    # 7. Legamenti di Cooper / Retinacula cutis con inserzioni subdermiche (VICTRE cooper.hxx)
    print("  * [7/10] Generazione legamenti di Cooper (retinacula cutis)...")
    cooper_mesh = generate_cooper_ligaments(tissue_mesh, skin_mesh, kinematics, num_ligaments=26)
    if cooper_mesh is not None:
        cooper_mesh = sanitize_mesh(cooper_mesh)
        scene.add_geometry(cooper_mesh, node_name="cooper_ligaments")
        print(f"    - Vertici: {len(cooper_mesh.vertices)}, Facce: {len(cooper_mesh.faces)}")

    # 8. Linfonodi intramammari reniformi (UOQ)
    print("  * [8/13] Generazione linfonodi intramammari reniformi (UOQ)...")
    lymph_mesh = generate_lymph_nodes(kinematics)
    lymph_mesh = sanitize_mesh(lymph_mesh)
    scene.add_geometry(lymph_mesh, node_name="lymph_nodes")

    # 9. Cuscino adiposo retromammario continuo (Spazio di Chassaignac)
    print("  * [9/13] Generazione cuscino adiposo retromammario continuo (spazio di Chassaignac)...")
    retro_mesh = generate_retromammary_fat(kinematics, skin_mesh)
    retro_mesh = sanitize_mesh(retro_mesh)
    scene.add_geometry(retro_mesh, node_name="retromammary_fat")

    # 10. Fascia pettorale profonda (Lamina profonda della fascia superficialis)
    print("  * [10/13] Generazione fascia pettorale profonda pre-muscolare...")
    fascia_mesh = generate_deep_pectoral_fascia(kinematics, skin_mesh)
    fascia_mesh = sanitize_mesh(fascia_mesh)
    scene.add_geometry(fascia_mesh, node_name="deep_pectoral_fascia")

    # 11. Muscolo Grande Pettorale (parete toracica posteriore)
    print("  * [11/13] Generazione muscolo pettorale maggiore (parete toracica)...")
    pec_mesh = generate_pectoral_muscle(kinematics, skin_mesh)
    pec_mesh = sanitize_mesh(pec_mesh)
    scene.add_geometry(pec_mesh, node_name="pectoral_muscle")
    print(f"    - Vertici: {len(pec_mesh.vertices)}, Facce: {len(pec_mesh.faces)}")

    # 12. Innervazione sensoriale cutanea (4° ramo cutaneo laterale del nervo intercostale + rami mediali)
    print("  * [12/13] Generazione nervi sensoriali cutanei (4° intercostale)...")
    nerves_mesh = generate_mammary_nerves(kinematics)
    nerves_mesh = sanitize_mesh(nerves_mesh)
    scene.add_geometry(nerves_mesh, node_name="mammary_nerves")

    # 13. Plesso linfatico subareolare di Sappey e vasi collettori ascellari
    print("  * [13/13] Generazione plesso linfatico di Sappey e vasi ascellari...")
    lymph_vessels_mesh = generate_lymphatic_vessels(kinematics)
    lymph_vessels_mesh = sanitize_mesh(lymph_vessels_mesh)
    scene.add_geometry(lymph_vessels_mesh, node_name="lymphatic_vessels")

    print("  * Nota: Lesione tumorale omessa su richiesta (verra ricostruita e posizionata separatamente).")
        
    # -----------------------------------------------------------------
    # FASE 3: Esportazione GLB e Scena Blender
    # -----------------------------------------------------------------
    print("\n[4/4] Esportazione modello 3D unificato multi-strato...")
    if not output_dir:
        output_dir = os.path.join(MODULE_06_DIR, "output", resolved_pid)
    os.makedirs(output_dir, exist_ok=True)
    
    out_glb = os.path.join(output_dir, f"{resolved_pid}_unified_3d.glb")
    out_blend = os.path.join(output_dir, f"{resolved_pid}_unified_3d.blend")
    out_preview = os.path.join(output_dir, f"{resolved_pid}_multi_layer_preview.png")
    out_vol_npy = os.path.join(output_dir, "breast_3d_volume.npy")

    # Salva il volume 3D neurale per ispezione
    np.save(out_vol_npy, volume_3d)

    with open(out_glb, "wb") as f:
        f.write(scene.export(file_type="glb"))
    print(f"  * File GLB multi-strato salvato: {out_glb}")
    print(f"  * Volume 3D U-Net-GAN salvato: {out_vol_npy}")
    
    # Se Blender è disponibile, genera il file .blend con illuminazione e shader
    default_blender = r"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe"
    if not blender_exe or not os.path.exists(blender_exe):
        blender_exe = default_blender if os.path.exists(default_blender) else "blender"
        
    if os.path.exists(blender_exe):
        setup_script = os.path.join(CURRENT_DIR, "blender_setup.py")
        import subprocess
        cmd = [blender_exe, "--background", "--python", setup_script, "--", out_glb, out_blend, out_preview]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(out_blend):
            print(f"  * File Scena Blender salvato: {out_blend}")
        if os.path.exists(out_preview):
            print(f"  * Rendering Preview salvato: {out_preview}")
            
    print("\n" + "=" * 75)
    print(" RICOSTRUZIONE UNIFICATA MULTI-STRATO COMPLETATA CON SUCCESSO!")
    print(f" Modello GLB: {out_glb}")
    if os.path.exists(out_blend):
        print(f" Blender 3D : {out_blend}")
    if os.path.exists(out_preview):
        print(f" Preview    : {out_preview}")
    print("=" * 75)
    
    return out_glb, out_blend


def main():
    parser = argparse.ArgumentParser(description="Ricostruzione 3D Unificata Seno + Lesione U-Net-GAN")
    parser.add_argument("--patient_id", type=str, default="P_00081", help="ID Paziente (es. P_00081, P_00001)")
    parser.add_argument("--data_dir", type=str, default=None, help="Cartella dati pazienti")
    parser.add_argument("--output_dir", type=str, default=None, help="Cartella di output")
    args = parser.parse_args()
    
    run_unified_pipeline(patient_id=args.patient_id, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
