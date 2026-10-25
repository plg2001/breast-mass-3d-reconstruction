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
├── 05_Ricostruzione_Mammella_3D/             # [SUPERFICIE MAMMELLA] Guscio 3D Bi-planare (Skin Envelope)
│   ├── breast_surface_reconstruction.py      # Algoritmo lofting quad-ellittico da CC ed MLO
│   ├── run_surface_reconstruction.bat        # Launcher automatico per ricostruzione e preview
│   ├── open_in_blender.bat                   # Apertura immediata del modello 3D e della scena
│   ├── output/                               # Mesh .obj, .glb, scena completa e rendering 6-viste
│   └── README.md                             # Documentazione geometrica e istruzioni
│
├── data/                                     # Dataset clinico mammografico (CBIS-DDSM)
├── Mammography3D.bat                         # Launcher principale rapido nella radice
└── README.md                                 # Questo file
```

---

## 🚀 Avvio Rapido

1. **Ricostruzione 3D Superficie Mammella (Guscio Cutaneo + Sezioni 2D)**:
   Entra in `05_Ricostruzione_Mammella_3D` ed esegui `run_surface_reconstruction.bat` oppure da terminale:
   ```powershell
   python 05_Ricostruzione_Mammella_3D/breast_surface_reconstruction.py --patient_id P_00005
   ```
   Genera il file OBJ/GLB della pelle, la scena con lastre semitrasparenti e la figura clinica a 6 viste.

2. **Ricostruzione 3D Sola Lesione**:
   * Modello SMRD (consigliato per realismo biologico):
     ```powershell
     python 04_Ricostruzione_Lesione_SMRD_Paper/smrd_lesion_reconstruction.py
     ```
   * Modello Ling Shao (PVC storico):
     ```powershell
     python 01_Ricostruzione_Lesione_Ling_Shao_Paper/lesion_pvc_reconstruction.py
     ```

3. **Applicazione Desktop (Mammography 3D)**:
   Fai doppio clic su `Mammography3D.bat` nella cartella principale.
