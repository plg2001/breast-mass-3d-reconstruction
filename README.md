# Progetto di Tesi: Ricostruzione 3D da Mammografie Bi-Planari

Organizzazione dell'area di lavoro, suddivisa per moduli indipendenti:

```
Tesi/
├── 01_Ricostruzione_Lesione_Ling_Shao_Paper/ # [BENCHMARK] Modello Ling Shao (2007 - PVC)
│   ├── data/                                 # Input: R-CC.jpg, R-MLO.jpg, ROI-R-CC.jpg, ROI-R-MLO.jpg
│   ├── output/                               # Modelli 3D (.glb, .blend, .obj) e grafici della lesione
│   ├── lesion_pvc_reconstruction.py          # Algoritmo matematico PVC Ling Shao
│   ├── run_reconstruction.bat                # Launcher per rieseguire la pipeline
│   ├── open_in_blender.bat                   # Apertura immediata della lesione in Blender
│   └── README.md                             # Dettagli matematici e benchmark del paper
│
├── 02_Applicazione_GUI_CSharp/               # [MODULO APPLICAZIONE] Interfaccia Grafica e Unity
│   ├── Mammography3D_Solution.sln            # Soluzione C#
│   ├── Mammography3D.Desktop/                # Codice sorgente GUI WPF (.NET 8)
│   ├── Mammography3D.Core/                   # Modelli, servizi e script pronti per Unity / VR
│   ├── Mammography3D_App/                    # Applicazione compilata pronta all'uso
│   └── Mammography3D.bat                     # Launcher locale dell'app
│
├── 03_Paper_e_Documenti/                     # [DOCUMENTAZIONE] Letteratura e proposta di tesi
│   ├── proposta_idea_tesi_visore_edge.pdf    # Proposta di tesi per visore Android / Edge
│   └── (Articoli scientifici PDF)
│
├── 04_Ricostruzione_Lesione_SMRD_Paper/      # [BENCHMARK] Modello SMRD (De Sisternes et al. 2015)
│   ├── smrd_lesion_reconstruction.py         # Algoritmo SMRD avanzato
│   ├── compare_ling_shao_vs_smrd.py          # Confronto quantitativo vs Ling Shao
│   ├── run_smrd.bat                          # Launcher pipeline SMRD
│   └── README.md                             # Dettagli matematici e benchmark clinico
│
├── 05_Ricostruzione_Superficie_Seno_Paper/   # [SUPERFICIE SENO] Ricostruzione Bi-planare (Huang / Pinto)
│   ├── main.py                               # Pipeline clinica di superficie
│   ├── kinematics.py                         # Allineamento bi-planare PNL e quote epipolari
│   ├── surface_reconstruction.py             # Lofting super-ellittico organico (curve di Lamé)
│   ├── segmentation.py                       # Segmentazione automatica bordo pelle
│   ├── run_reconstruction.bat                # Launcher rapido per superficie cutanea
│   ├── open_in_blender.bat                   # Apertura immediata della superficie in Blender
│   └── README.md                             # Documentazione teorica e paper
│
├── 06_U-Net_GAN/                             # [DEEP LEARNING + FDA VICTRE] Modello 3D Multi-Strato
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
├── data/                                     # Dataset clinico mammografico (CBIS-DDSM)
├── Mammography3D.bat                         # Launcher principale rapido nella radice
└── README.md                                 # Questo file
```

---

## 🚀 Avvio Rapido

1. **Ricostruzione 3D Anatomica Completa a 14 Strati (3D U-Net-GAN + FDA VICTRE)**:
   Entra in `06_U-Net_GAN` ed esegui `run_inference.bat` oppure da terminale:
   ```powershell
   python 06_U-Net_GAN/src/unified_reconstruction.py --patient_id P_00081
   ```
   Genera il modello 3D completo a 14 strati (`.glb`), la scena Blender (`.blend`) con materiali semitrasparenti PBR/SSS e il volume neurale (`breast_3d_volume.npy`).

2. **Ricostruzione 3D Superficie Seno (Sola Cute Esterna)**:
   Entra in `05_Ricostruzione_Superficie_Seno_Paper` ed esegui `run_reconstruction.bat` oppure da terminale:
   ```powershell
   python 05_Ricostruzione_Superficie_Seno_Paper/main.py --patient_id P_00081
   ```

3. **Ricostruzione 3D Sola Lesione Tumorale**:
   * Modello SMRD (consigliato per realismo biologico):
     ```powershell
     python 04_Ricostruzione_Lesione_SMRD_Paper/smrd_lesion_reconstruction.py
     ```
   * Modello Ling Shao (PVC storico):
     ```powershell
     python 01_Ricostruzione_Lesione_Ling_Shao_Paper/lesion_pvc_reconstruction.py
     ```

4. **Applicazione Desktop (Mammography 3D)**:
   Fai doppio clic su `Mammography3D.bat` nella cartella principale.
