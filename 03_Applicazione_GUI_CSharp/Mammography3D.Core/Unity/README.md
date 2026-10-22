# Integrazione Unity per Visore VR (Meta Quest / Android / PCVR)

Questa cartella contiene i componenti C# pronti all'uso per Unity, sviluppati specificamente per l'architettura descritta nella proposta di tesi:
> **"Applicazione immersiva Android supportata da Edge/Fog Computing per la visualizzazione 3D di lesioni mammografiche"**

---

## 📦 Come importare nel tuo progetto Unity

1. Copia i file `.cs` di `Mammography3D.Core` (compresi `Models/`, `Services/` e `Unity/`) all'interno della cartella `Assets/Scripts/Mammography3D/` del tuo progetto Unity.
2. In Unity, apri il **Package Manager** e installa:
   - **`glTFast`** (oppure `UnityGLTF`) per caricare i file `.glb` a runtime.
   - **`XR Interaction Toolkit`** / **`Meta OpenXR`** (se stai sviluppando per visori Android come Meta Quest 2 / 3 / Pro).

---

## 🚀 Utilizzo in Scena

1. Trascina lo script `MammographyReconstructionBridge` su un GameObject vuoto nella tua scena.
2. Nell'Inspector puoi scegliere:
   - **Modalità Locale (`useEdgeServer = false`)**: Esegue direttamente la pipeline Python (ottimale per test in Unity Editor o build Windows PCVR).
   - **Modalità Edge Server (`useEdgeServer = true`)**: Invia le immagini al server Edge via REST API e riceve indietro il modello 3D `.glb` con le relative metriche (indispensabile per visori standalone Android).
3. Assegna l'anchor di spawn (`lesionSpawnAnchor`) dove vuoi che compaia il modello anatomico.
4. Collega l'evento `OnReconstructionSuccess` per mostrare il pannello annotazioni `LesionAnnotationMarker` con volume, sfericità e predizioni.
