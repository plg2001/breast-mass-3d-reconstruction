# Progetto di Tesi: Ricostruzione 3D da Mammografie Bi-Planari

Organizzazione dell'area di lavoro, suddivisa per moduli indipendenti:

```
Tesi/
│
├── 01_Ricostruzione_Lesione_3D/       # [MODULO ATTIVO] Ricostruzione 3D della SOLA Lesione (Ling Shao 2007 - PVC)
│   ├── data/                          # Input: R-CC.jpg, R-MLO.jpg, ROI-R-CC.jpg, ROI-R-MLO.jpg
│   ├── output/                        # Modelli 3D (.glb, .blend, .obj) e grafici della sola lesione
│   ├── lesion_pvc_reconstruction.py   # Algoritmo matematico puro per la lesione
│   ├── run_reconstruction.bat         # Launcher per rieseguire la pipeline della lesione
│   ├── open_in_blender.bat            # Apertura immediata della lesione in Blender
│   └── README.md                      # Dettagli matematici e benchmark del paper
│
├── 02_Ricostruzione_Mammella_3D/      # [MODULO SEPARATO] Ricostruzione 3D della Mammella (Precedente)
│   ├── breast_reconstruction.py       # Algoritmo per la sagoma volumetrica della mammella
│   └── README.md                      # Note: modulo isolato, NON integrato con la lesione
│
├── 03_Applicazione_GUI_CSharp/        # [MODULO APPLICAZIONE] Interfaccia Grafica e Unity
│   ├── Mammography3D_Solution.sln     # Soluzione C#
│   ├── Mammography3D.Desktop/         # Codice sorgente GUI WPF (.NET 8)
│   ├── Mammography3D.Core/            # Modelli, servizi e script pronti per Unity / VR
│   ├── Mammography3D_App/             # Applicazione compilata pronta all'uso
│   └── Avvia_Interfaccia_Grafica.bat  # Launcher locale dell'app
│
├── 04_Paper_e_Documenti/              # [DOCUMENTAZIONE] Letteratura e proposta di tesi
│   ├── proposta_idea_tesi_visore_edge.pdf # Proposta di tesi per visore Android / Edge
│   └── (Articoli scientifici PDF)
│
├── Avvia_Interfaccia_Grafica.bat      # 🚀 Launcher principale rapido nella radice
└── README.md                          # Questo file
```

---

## ⚡ Avvio Rapido

1. **Interfaccia Grafica C#**:
   Fai doppio clic su `Avvia_Interfaccia_Grafica.bat` nella cartella principale.
   L'app consente di selezionare le immagini (con anteprime e drag-and-drop), avviare la ricostruzione in background e aprire il modello in Blender.

2. **Esecuzione diretta da Script Python (Sola Lesione)**:
   Entra in `01_Ricostruzione_Lesione_3D` e fai doppio clic su `run_reconstruction.bat` oppure esegui:
   ```powershell
   python 01_Ricostruzione_Lesione_3D/lesion_pvc_reconstruction.py
   ```
