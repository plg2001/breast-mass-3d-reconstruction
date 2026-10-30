"""
Inferenza con il modello 3D U-Net-GAN per la ricostruzione 3D di mammelle.

Uso:
    python inference.py --weights "training result/unet3d_generator.pth" --input image.jpg --output breast_3d.glb
    python inference.py --weights "training result/unet3d_generator.pth" --synthetic --output breast_3d.glb

Produce un file GLB con:
- Mesh della mammella (tessuto semi-trasparente)
- Mesh della lesione (rosso solido)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import scipy.ndimage as ndimage

import torch
import torch.nn.functional as F

from skimage.measure import marching_cubes
import trimesh

from config import Config
from model import UNet3DGenerator
from dataset import build_seed_and_target_volumes, generate_synthetic_mammogram

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(weights_path: str) -> UNet3DGenerator:
    """Carica il generatore con i pesi pre-addestrati."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"File pesi non trovato: {weights_path}")
    
    generator = UNet3DGenerator().to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    generator.load_state_dict(state_dict)
    generator.eval()
    print(f"Modello caricato da: {weights_path}")
    return generator


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pipeline clinica di preprocessing per mammografie 2D:
    1. Rimozione margini dello scanner / digitalizzatore (2% del perimetro azzerato).
    2. Segmentazione mammella: estrazione della componente connessa principale
       ed eliminazione di etichette di testo ("L MLO", "R CC", marker radio-opachi).
    3. Ritaglio (Bounding Box crop) della mammella per eliminare lo spazio nero
       ed evitare che il seno venga deformato/schiacciato nel resize.
    4. Rilevamento della parete toracica e allineamento (parete a sinistra, cupola a destra).
    5. Normalizzazione fotometrica calcolata esclusivamente sui pixel del tessuto.
    6. Ridimensionamento a [128, 128] normalizzato in [0, 1].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")

    h, w = img.shape

    # 1. Pulizia bordi scanner / fotogramma (azzeramento perimetrale del 2%)
    margin_y = max(10, int(0.02 * h))
    margin_x = max(10, int(0.02 * w))
    cleaned = img.copy()
    cleaned[:margin_y, :] = 0
    cleaned[h - margin_y:, :] = 0
    cleaned[:, :margin_x] = 0
    cleaned[:, w - margin_x:] = 0

    # 2. Segmentazione mammella: ricerca della componente connessa più grande
    target_h = 600
    scale = target_h / float(h)
    target_w = max(1, int(w * scale))
    small = cv2.resize(cleaned, (target_w, target_h), interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(small, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_small = (labels == largest_label).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        masked_img = np.where(mask > 0, cleaned, 0)
    else:
        masked_img = cleaned
        mask = (cleaned > 15).astype(np.uint8) * 255

    # 3. Bounding box attorno alla mammella (elimina il 40% di aria nera circostante)
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) > 0 and len(x_indices) > 0:
        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()
        pad_y = int(0.02 * (ymax - ymin))
        pad_x = int(0.02 * (xmax - xmin))
        ymin = max(0, ymin - pad_y)
        ymax = min(h - 1, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(w - 1, xmax + pad_x)
        cropped = masked_img[ymin:ymax+1, xmin:xmax+1]
    else:
        cropped = masked_img

    # 4. Rilevamento parete toracica e allineamento (modello addestrato con torace a sinistra)
    ch, cw = cropped.shape
    border_zone = max(5, int(0.12 * cw))
    left_density = float(cropped[:, :border_zone].mean())
    right_density = float(cropped[:, -border_zone:].mean())
    if right_density > left_density:
        cropped = cv2.flip(cropped, 1)

    # 5. Normalizzazione del contrasto calcolata solo sui pixel del tessuto
    tissue_pixels = cropped[cropped > 0]
    if len(tissue_pixels) > 50:
        p1, p99 = np.percentile(tissue_pixels, (1, 99))
        if p99 > p1:
            cropped_float = np.clip((cropped.astype(np.float32) - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
            cropped_float[cropped == 0] = 0.0
        else:
            cropped_float = cropped.astype(np.float32) / 255.0
    else:
        cropped_float = cropped.astype(np.float32) / 255.0

    # 6. Resize anatomico a 128x128
    img_resized = cv2.resize(cropped_float, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA)

    return np.clip(img_resized, 0.0, 1.0).astype(np.float32)


def run_inference(generator: UNet3DGenerator, img_2d: np.ndarray) -> np.ndarray:
    """
    Esegue l'inferenza: data una mammografia 2D, produce il volume 3D.
    
    Args:
        generator: Il modello generatore caricato
        img_2d: Array numpy [128, 128] normalizzato in [0, 1]
    
    Returns:
        volume_3d: Array numpy [10, 128, 128] in [0, 1]
    """
    seed_vol, _ = build_seed_and_target_volumes(img_2d)
    seed_tensor = torch.from_numpy(seed_vol[None, ...]).to(DEVICE)  # [1, 1, 10, 128, 128]
    
    with torch.no_grad():
        pred_vol = generator(seed_tensor)  # [1, 1, 10, 128, 128]
    
    volume = pred_vol[0, 0].cpu().numpy()  # [10, 128, 128]
    return volume


def volume_to_mesh(volume: np.ndarray, threshold: float, spacing: tuple) -> trimesh.Trimesh:
    """
    Estrae una mesh da un volume 3D tramite Marching Cubes.
    
    Args:
        volume: Array numpy [D, H, W] con valori in [0, 1]
        threshold: Soglia isosurface
        spacing: (sz, sy, sx) spaziatura in mm per ogni asse
    
    Returns:
        mesh: trimesh.Trimesh oppure None se non trovata
    """
    if volume.max() <= threshold:
        return None
    
    try:
        verts, faces, normals, _ = marching_cubes(volume, level=threshold, spacing=spacing)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
        return mesh
    except Exception as e:
        print(f"  Errore marching cubes (soglia={threshold:.2f}): {e}")
        return None


def safe_smooth_mesh(mesh: trimesh.Trimesh, lamb: float = 0.5, iterations: int = 3):
    """
    Applica smoothing laplaciano in modo sicuro.
    IMPORTANTE: volume_constraint=False per evitare divisioni per zero o volumi negativi
    che generano NaN nei vertici glTF / GLB.
    """
    if mesh is None or len(mesh.vertices) == 0:
        return
    try:
        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=lamb,
            iterations=iterations,
            volume_constraint=False,
            implicit_time_integration=False
        )
    except Exception as e:
        print(f"    Avviso smoothing: {e}")

    # Pulizia facce degenerate e vertici non referenziati
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Verifica finale assenza di NaN / Inf
    if np.isnan(mesh.vertices).any() or np.isinf(mesh.vertices).any():
        mesh.vertices = np.nan_to_num(mesh.vertices, nan=0.0, posinf=0.0, neginf=0.0)
    if mesh.vertex_normals is not None and (np.isnan(mesh.vertex_normals).any() or np.isinf(mesh.vertex_normals).any()):
        mesh.vertex_normals = np.nan_to_num(mesh.vertex_normals, nan=0.0, posinf=0.0, neginf=0.0)


def create_glb(volume_3d: np.ndarray, output_path: str, include_parenchyma: bool = True):
    """
    Crea un file GLB anatomico e pulito con gerarchia a strati:
      1. Superficie esterna (Pelle / Breast Shell): guscio liscio convesso (rosa chiaro trasparente)
      2. Tessuto fibroglandolare interno (Parenchima): tessuto ghiandolare intermedio (opzionale, arancio/rosa tenue)
      3. Lesione tumorale: massa ad alta densità (rosso solido)
    """
    # Spaziatura fisica in mm: (profondità Z, altezza Y, larghezza X)
    # SLICE_INTERVAL = 5.0 mm per dare la corretta proporzione volumetrica dello spessore
    spacing = (Config.SLICE_INTERVAL, Config.S_XY, Config.S_XY)
    
    scene = trimesh.Scene()
    meshes_added = 0
    
    # -------------------------------------------------------------
    # 1. Superficie esterna della mammella (Guscio / Cute)
    # -------------------------------------------------------------
    print("  [1/3] Estrazione superficie esterna (pelle liscia)...")
    # Troviamo la sagoma complessiva (qualsiasi tessuto > 0.03)
    breast_mask = volume_3d > 0.03
    # Riempiamo i vuoti interni slice per slice per evitare buchi all'interno della pelle
    filled_mask = np.zeros_like(breast_mask, dtype=bool)
    for d in range(breast_mask.shape[0]):
        filled_mask[d] = ndimage.binary_fill_holes(breast_mask[d])
    
    # Chiudiamo eventuali imperfezioni 3D e lisciamo con filtro gaussiano
    filled_mask = ndimage.binary_closing(filled_mask, structure=np.ones((2, 3, 3)))
    smoothed_shell = ndimage.gaussian_filter(filled_mask.astype(np.float32), sigma=(0.8, 1.2, 1.2))
    
    skin_mesh = volume_to_mesh(smoothed_shell, threshold=0.45, spacing=spacing)
    if skin_mesh is not None and len(skin_mesh.vertices) > 0:
        safe_smooth_mesh(skin_mesh, lamb=0.5, iterations=4)
        # Rosa tenue / carne semitrasparente (RGBA: 255, 205, 195, 75)
        skin_color = np.array([255, 205, 195, 75], dtype=np.uint8)
        skin_mesh.visual.vertex_colors = np.tile(skin_color, (len(skin_mesh.vertices), 1))
        scene.add_geometry(skin_mesh, node_name="breast_skin")
        meshes_added += 1
        print(f"    Superficie esterna: {len(skin_mesh.vertices)} vertici, {len(skin_mesh.faces)} facce")
    
    # -------------------------------------------------------------
    # 2. Tessuto fibroglandolare interno (Parenchima)
    # -------------------------------------------------------------
    if include_parenchyma:
        print("  [2/3] Estrazione tessuto ghiandolare interno...")
        parenchyma_threshold = 0.35
        # Smussiamo il volume per evitare artefatti ad alta frequenza
        smoothed_vol = ndimage.gaussian_filter(volume_3d, sigma=(0.6, 0.8, 0.8))
        tissue_mesh = volume_to_mesh(smoothed_vol, threshold=parenchyma_threshold, spacing=spacing)
        
        if tissue_mesh is not None and len(tissue_mesh.vertices) > 0:
            safe_smooth_mesh(tissue_mesh, lamb=0.4, iterations=2)
            # Arancio/giallo ghiandolare semitrasparente (RGBA: 240, 175, 120, 110)
            tissue_color = np.array([240, 175, 120, 110], dtype=np.uint8)
            tissue_mesh.visual.vertex_colors = np.tile(tissue_color, (len(tissue_mesh.vertices), 1))
            scene.add_geometry(tissue_mesh, node_name="fibroglandular_tissue")
            meshes_added += 1
            print(f"    Tessuto ghiandolare: {len(tissue_mesh.vertices)} vertici, {len(tissue_mesh.faces)} facce")
    
    # -------------------------------------------------------------
    # 3. Lesione tumorale (Massa iperintensa)
    # -------------------------------------------------------------
    print("  [3/3] Estrazione lesione tumorale...")
    lesion_threshold = Config.ROI_THRESHOLD  # 0.70
    lesion_vol = ndimage.gaussian_filter(volume_3d, sigma=(0.5, 0.6, 0.6))
    lesion_mesh = volume_to_mesh(lesion_vol, threshold=lesion_threshold, spacing=spacing)
    
    if lesion_mesh is not None and len(lesion_mesh.vertices) > 0:
        safe_smooth_mesh(lesion_mesh, lamb=0.5, iterations=2)
        # Rosso intenso opaco (RGBA: 220, 30, 30, 255)
        lesion_color = np.array([220, 30, 30, 255], dtype=np.uint8)
        lesion_mesh.visual.vertex_colors = np.tile(lesion_color, (len(lesion_mesh.vertices), 1))
        scene.add_geometry(lesion_mesh, node_name="tumor_lesion")
        meshes_added += 1
        print(f"    Lesione: {len(lesion_mesh.vertices)} vertici, {len(lesion_mesh.faces)} facce")
    else:
        print(f"    Nessuna lesione oltre la soglia {lesion_threshold}")
    
    if meshes_added == 0:
        placeholder = trimesh.creation.box(extents=[1, 1, 1])
        scene.add_geometry(placeholder, node_name="placeholder")
    
    # Esportazione GLB
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb_data = scene.export(file_type="glb")
    with open(output_path, "wb") as f:
        f.write(glb_data)
    
    print(f"  File GLB salvato: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")


def resolve_data_directory(data_dir: str = None) -> str:
    """Risolve il percorso della cartella data."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        data_dir,
        os.path.join(script_dir, "..", "data"),
        os.path.join(script_dir, "..", "..", "data"),
        os.path.join(script_dir, "data"),
        "data",
        "../data",
        "../../data",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isdir(c):
            return os.path.abspath(c)
    return None


def resolve_patient_image(data_dir: str, patient_id: str, view: str = "AUTO") -> tuple:
    """
    Risolve l'ID del paziente e trova l'immagine mammografica 2D corrispondente.
    Ritorna: (image_path, resolved_pid, resolved_view)
    """
    clean = patient_id.strip()
    digits = ''.join(c for c in clean if c.isdigit())
    
    candidates_pid = [clean]
    if digits:
        candidates_pid.append(f"P_{int(digits):05d}")
        candidates_pid.append(f"P_{digits}")
        candidates_pid.append(digits)
    
    patient_dir = None
    resolved_pid = clean
    for pid in candidates_pid:
        p_dir = os.path.join(data_dir, pid)
        if os.path.exists(p_dir) and os.path.isdir(p_dir):
            patient_dir = p_dir
            resolved_pid = pid
            break
            
    if patient_dir is None:
        raise FileNotFoundError(f"Cartella paziente non trovata per ID '{patient_id}' in: {data_dir}")
        
    # Elenco immagini escludendo quelle di ROI o maschere
    all_files = [f for f in os.listdir(patient_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')) and not f.upper().startswith("ROI-")]
    
    if not all_files:
        raise FileNotFoundError(f"Nessuna immagine mammografica trovata in: {patient_dir}")
        
    view_upper = view.upper()
    selected_file = None
    selected_view = "UNKNOWN"
    
    if view_upper in ("MLO", "AUTO"):
        for f in all_files:
            if "MLO" in f.upper():
                selected_file = f
                selected_view = "MLO"
                break
                
    if selected_file is None and view_upper in ("CC", "AUTO"):
        for f in all_files:
            if "CC" in f.upper():
                selected_file = f
                selected_view = "CC"
                break
                
    if selected_file is None:
        selected_file = all_files[0]
        selected_view = "VIEW"
        
    return os.path.join(patient_dir, selected_file), resolved_pid, selected_view


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferenza U-Net-GAN: mammografia 2D -> volume 3D -> GLB"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "training result", "unet3d_generator.pth"),
        help="Percorso al file dei pesi del generatore (.pth)"
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="ID del paziente (es. P_00081 o 81). Cerca automaticamente l'immagine in data_dir"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Percorso alla cartella 'data' contenente i pazienti"
    )
    parser.add_argument(
        "--view",
        type=str,
        default="AUTO",
        choices=["AUTO", "MLO", "CC"],
        help="Vista mammografica da utilizzare (AUTO: preferisce MLO, altrimenti CC)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Percorso diretto all'immagine mammografica 2D in input (jpg/png/pgm)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Usa una mammografia sintetica invece di un file in input"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Percorso del file GLB di output (default: calcolato in output/[paziente]/)"
    )
    parser.add_argument(
        "--save_volume",
        action="store_true",
        help="Salva anche il volume 3D come file numpy (.npy)"
    )
    parser.add_argument(
        "--no_parenchyma",
        action="store_true",
        help="Esclude il tessuto ghiandolare interno (mostra solo pelle esterna + lesione)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("  3D Breast Mammogram Reconstruction - Inferenza")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    
    # 1. Carica il modello
    print("\n[1/4] Caricamento modello...")
    generator = load_generator(args.weights)
    
    # 2. Carica/genera l'immagine 2D
    print("\n[2/4] Preparazione immagine 2D...")
    selected_image_path = args.input
    patient_id_label = "SYNTHETIC"
    view_label = "SYNTHETIC"
    
    if args.patient_id:
        data_dir = resolve_data_directory(args.data_dir)
        if not data_dir:
            raise FileNotFoundError("Cartella 'data' non trovata. Specifica --data_dir.")
        selected_image_path, patient_id_label, view_label = resolve_patient_image(
            data_dir, args.patient_id, args.view
        )
        print(f"  Paziente selezionato: {patient_id_label}")
        print(f"  Vista selezionata   : {view_label}")
        print(f"  Immagine trovata    : {selected_image_path}")
        
    if selected_image_path:
        print(f"  Caricamento da: {selected_image_path}")
        img_2d = preprocess_image(selected_image_path)
        print(f"  Immagine preprocessata: {img_2d.shape}, range [{img_2d.min():.3f}, {img_2d.max():.3f}]")
    else:
        print("  Generazione mammografia sintetica...")
        raw_img = generate_synthetic_mammogram(Config.IMG_SIZE)
        img_2d = raw_img.astype(np.float32) / 255.0
        imin, imax = img_2d.min(), img_2d.max()
        if imax - imin > 1e-6:
            img_2d = (img_2d - imin) / (imax - imin)
        img_2d = np.clip(img_2d, 0.0, 1.0).astype(np.float32)
        print(f"  Immagine sintetica generata: {img_2d.shape}, range [{img_2d.min():.3f}, {img_2d.max():.3f}]")

    # Determina il percorso di output finale
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_folder = os.path.dirname(script_dir)  # Cartella 07_U-Net_GAN
    
    if args.output:
        final_output = args.output
    elif args.patient_id:
        final_output = os.path.join(project_folder, "output", patient_id_label, f"{patient_id_label}_{view_label}_3d.glb")
    else:
        final_output = os.path.join(project_folder, "output", "synthetic", "breast_3d.glb")
    final_output = os.path.abspath(final_output)
    
    # 3. Inferenza
    print("\n[3/4] Inferenza (generazione volume 3D)...")
    volume_3d = run_inference(generator, img_2d)
    print(f"  Volume generato: {volume_3d.shape}, range [{volume_3d.min():.3f}, {volume_3d.max():.3f}]")
    print(f"  Voxel > 0.7 (potenziale lesione): {(volume_3d > 0.7).sum()}")
    
    if args.save_volume:
        vol_path = final_output.replace(".glb", "_volume.npy")
        os.makedirs(os.path.dirname(os.path.abspath(vol_path)), exist_ok=True)
        np.save(vol_path, volume_3d)
        print(f"  Volume 3D salvato: {vol_path}")
    
    # 4. Crea il file GLB
    print("\n[4/4] Creazione file GLB...")
    create_glb(volume_3d, final_output, include_parenchyma=not args.no_parenchyma)
    
    print("\n" + "=" * 60)
    print(f"  Output: {final_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
