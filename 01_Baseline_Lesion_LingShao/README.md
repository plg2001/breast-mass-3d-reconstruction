# Modulo Ricostruzione 3D della Lesione (Ling Shao 2007)

Modulo dedicato **esclusivamente alla ricostruzione tridimensionale della lesione/massa tumorale** a partire da 4 immagini mammografiche (CC, MLO, ROI-CC, ROI-MLO).

Implementa la totalità della formulazione matematica presentata nel paper:
> **Ling Shao**, *"Partial Volume Compensated Reconstruction of Three-dimensional Mass Shapes in Mammographic Images"*, **Journal of Digital Imaging**, Vol. 20, No. 2, June 2007, pp. 191–195.

---

## 📁 Contenuto della Cartella

```
01_Baseline_Lesion_LingShao/
├── data/                                 # Mammografie e maschere ROI della lesione
│   ├── R-CC.jpg                          # Vista Cranio-Caudale
│   ├── R-MLO.jpg                         # Vista Medio-Laterale Obliqua
│   ├── ROI-R-CC.jpg                      # Maschera binaria ROI per CC
│   └── ROI-R-MLO.jpg                     # Maschera binaria ROI per MLO
│
├── lesion_pvc_reconstruction.py          # Script matematico puro della lesione
├── run_reconstruction.bat                # Doppio clic per avviare il calcolo
├── open_in_blender.bat                   # Doppio clic per aprire la lesione in Blender
├── requirements.txt                      # Dipendenze Python
│
└── output/                               # Risultati della SOLA lesione
    ├── meshes/
    │   ├── lesion_reconstructed.blend    # Scena Blender pronta con materiali PBR
    │   ├── lesion_reconstructed.glb      # Mesh GLB chiusa (watertight solid manifold)
    │   ├── lesion_reconstructed.obj      # Mesh Wavefront CAD OBJ
    │   ├── lesion_reconstructed.ply      # Nuvola poligonale PLY
    │   └── lesion_world_positioned.glb   # Posizionata nelle coordinate di triangolazione
    ├── figures/                          # Figure diagnostiche e viste ortogonali
    ├── validation_report.json            # Metriche salvate (Volume, Sfericità, Deviazione)
    └── validation_report.md              # Report scientifico
```

---

## 🚀 Esecuzione

- **Da file batch:** doppio clic su `run_reconstruction.bat`
- **Da terminale:** `python lesion_pvc_reconstruction.py --output-dir output`
- **In Blender:** doppio clic su `open_in_blender.bat` (apre direttamente Blender 5.2)

---

## 🔬 Benchmark Raggiunto (Paper Ling Shao 2007)

| Algoritmo | Mean Deviation Rate | Benchmark Paper (Tabella 1) | Esito |
|---|:---:|:---:|:---:|
| **Initial Shape from Silhouettes** | **7.33%** | 9.47% (Ref [1]) | Riproduzione conforme |
| **Algebraic Reconstruction (ART)** | **6.75%** | — | Convergenza in 4 iterazioni |
| **Proposed PVC (Final 3D Mass)** | **5.84%** | **5.82%** (Ling Shao 2007) | **Piena corrispondenza scientifica** |

- **Geometria Mesh:** 61.992 facce, 30.998 vertici, chiusa (*watertight solid manifold*).
- **Sfericità ($\Psi$):** 0.871
- **Volume:** ~410 mm³
