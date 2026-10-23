# Modulo Interfaccia Grafica C# e Supporto Unity

Questo modulo contiene:
1. **L'applicazione Desktop WPF (.NET 8)** per la selezione delle immagini mammografiche e la ricostruzione 3D della lesione in background.
2. **La libreria Core C# e i componenti per Unity / Visore VR** per il caricamento e l'interazione con il modello 3D della lesione.

---

## 📁 Struttura

- **`Mammography3D.Desktop/`**: Codice sorgente dell'interfaccia grafica WPF.
  - 4 Box di selezione immagini (R-CC, R-MLO, ROI-CC, ROI-MLO) con anteprime e Drag & Drop.
  - Regolazione parametri: angolo MLO, risoluzione griglia voxel.
  - Esecuzione asincrona in background con barra di avanzamento e log in tempo reale.
  - Visualizzazione metriche scientifiche (Volume mm³, Sfericità, Dice score, Deviazione).
  - Pulsante diretto per aprire la lesione in Blender (`C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe`).
- **`Mammography3D.Core/`**: Libreria compatibile al 100% con Unity (.NET Standard 2.0).
  - `Models/`: Modelli dati per configurazione, avanzamento e risultati.
  - `Services/`: Servizio locale Python e servizio HTTP per nodo Edge/Fog.
  - `Unity/`: Script pronti all'uso per Unity (`MammographyReconstructionBridge.cs`, `LesionAnnotationMarker.cs`).
- **`Mammography3D_App/`**: Eseguibile compilato standalone pronto all'avvio (`Mammography3D.Desktop.exe`).

---

## 🚀 Avvio Rapido

- Fai doppio clic su **`Mammography3D.bat`**.
