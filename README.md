# Progetto di Tesi: Ricostruzione 3D da Mammografie Bi-Planari

Organizzazione dell'area di lavoro, suddivisa per moduli indipendenti:

```
Tesi/
├── 01_Baseline_Lesion_LingShao/ # [BENCHMARK] Modello Ling Shao (2007 - PVC)
│   ├── data/                                 # Input: R-CC.jpg, R-MLO.jpg, ROI-R-CC.jpg, ROI-R-MLO.jpg
│   ├── output/                               # Modelli 3D (.glb, .blend, .obj) e grafici della lesione
│   ├── lesion_pvc_reconstruction.py          # Algoritmo matematico PVC Ling Shao
│   ├── run_reconstruction.bat                # Launcher per rieseguire la pipeline
│   ├── open_in_blender.bat                   # Apertura immediata della lesione in Blender
│   └── README.md                             # Dettagli matematici e benchmark del paper
│
├── 02_Desktop_Client_WPF/               # [MODULO APPLICAZIONE] Interfaccia Grafica e Unity
│   ├── Mammography3D_Solution.sln            # Soluzione C#
│   ├── Mammography3D.Desktop/                # Codice sorgente GUI WPF (.NET 8)
│   ├── Mammography3D.Core/                   # Modelli, servizi e script pronti per Unity / VR
│   ├── Mammography3D_App/                    # Applicazione compilata pronta all'uso
│   └── Mammography3D.bat                     # Launcher locale dell'app
│
├── 03_Scientific_Literature_and_Docs/                     # [DOCUMENTAZIONE] Letteratura e proposta di tesi
│   ├── proposta_idea_tesi_visore_edge.pdf    # Proposta di tesi per visore Android / Edge
│   └── (Articoli scientifici PDF)
│
├── 04_Spicular_Lesion_SMRD_Module/      # [BENCHMARK] Modello SMRD (De Sisternes et al. 2015)
│   ├── smrd_lesion_reconstruction.py         # Algoritmo SMRD avanzato
│   ├── compare_ling_shao_vs_smrd.py          # Confronto quantitativo vs Ling Shao
│   ├── run_smrd.bat                          # Launcher pipeline SMRD
│   └── README.md                             # Dettagli matematici e benchmark clinico
│
├── 05_Breast_Surface_Module/   # [SUPERFICIE SENO] Ricostruzione Bi-planare (Huang / Pinto)
│   ├── main.py                               # Pipeline clinica di superficie
│   ├── kinematics.py                         # Allineamento bi-planare PNL e quote epipolari
│   ├── surface_reconstruction.py             # Lofting super-ellittico organico (curve di Lamé)
│   ├── segmentation.py                       # Segmentazione automatica bordo pelle
│   ├── run_reconstruction.bat                # Launcher rapido per superficie cutanea
│   ├── open_in_blender.bat                   # Apertura immediata della superficie in Blender
│   └── README.md                             # Documentazione teorica e paper
│
├── 06_Deep_Anatomy_AI_Module/                             # [DEEP LEARNING + FDA VICTRE] Modello 3D Multi-Strato
│   ├── src/
│   │   ├── unified_reconstruction.py         # Pipeline unificata 14 strati anatomici
│   │   ├── victre_anatomy.py                 # Generatore bio-funzionale FDA VICTRE (dotti, vene, nervi, ecc.)
│   │   ├── blender_setup.py                  # Setup scena Blender con shader PBR e SSS
│   │   ├── model.py                          # Architettura 3D U-Net Generatore + Discriminatore GAN
│   │   └── inference.py                      # Inferenza della densità volumetrica da lastre 2D
│   ├── training result/                      # Pesi pre-addestrati unet3d_generator.pth
│   ├── run_inference.bat                     # Launcher completo per ricostruzione multi-strato
│   └── output/                               # Modelli .glb, scene .blend e volumi 3D .npy
│
├── 07_Total_Reconstruction_Module/ # [INTEGRAZIONE TOTALE] Mammella a 14 Strati + Tumore SMRD
│   ├── src/
│   │   ├── total_reconstruction.py           # Pipeline unificata totale (Mammella + Tumore localizzato)
│   │   ├── localization_engine.py            # Triangolazione epipolare stereotassica 3D (Kita / Huang)
│   │   ├── smrd_adapter.py                   # Integrazione Spicule Engine SMRD (De Sisternes et al.)
│   │   ├── victre_anatomy.py                 # Generatore 14 strati biologici completi
│   │   └── blender_total_setup.py            # Setup scena Blender a 15 oggetti PBR
│   ├── run_total_reconstruction.bat          # Launcher per la ricostruzione totale
│   ├── open_in_blender.bat                   # Apertura immediata della scena totale in Blender
│   └── README.md                             # Documentazione scientifica e manuale clinico
│
├── data/                                     # Dataset clinico mammografico (CBIS-DDSM)
├── Mammography3D.bat                         # Launcher principale rapido nella radice
└── README.md                                 # Questo file
```

---

## 🚀 Avvio Rapido

1. **Ricostruzione 3D Totale: Mammella a 14 Strati + Lesione Tumorale SMRD**:
   Entra in `07_Total_Reconstruction_Module` ed esegui `run_total_reconstruction.bat` oppure da terminale:
   ```powershell
   python 07_Total_Reconstruction_Module/src/total_reconstruction.py --patient_id P_00081
   ```
   Genera il modello 3D totale a 15 elementi (`.glb`), la scena Blender fotorealistica (`.blend`) e il report clinico completo (`_total_clinical_report.json`).

2. **Ricostruzione 3D Anatomica della Mammella (14 Strati senza lesione)**:
   Entra in `06_Deep_Anatomy_AI_Module` ed esegui `run_inference.bat` oppure:
   ```powershell
   python 06_Deep_Anatomy_AI_Module/src/unified_reconstruction.py --patient_id P_00081
   ```

2. **Ricostruzione 3D Superficie Seno (Sola Cute Esterna)**:
   Entra in `05_Breast_Surface_Module` ed esegui `run_reconstruction.bat` oppure da terminale:
   ```powershell
   python 05_Breast_Surface_Module/main.py --patient_id P_00081
   ```

3. **Ricostruzione 3D Sola Lesione Tumorale**:
   * Modello SMRD (consigliato per realismo biologico):
     ```powershell
     python 04_Spicular_Lesion_SMRD_Module/smrd_lesion_reconstruction.py
     ```
   * Modello Ling Shao (PVC storico):
     ```powershell
     python 01_Baseline_Lesion_LingShao/lesion_pvc_reconstruction.py
     ```

4. **Applicazione Desktop (Mammography 3D)**:
   Fai doppio clic su `Mammography3D.bat` nella cartella principale.
