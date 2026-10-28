# Modulo 06: Ricostruzione 3D della Superficie del Seno da Mammografie 2D (Paper Huang / Pinto / Kita)

Questo modulo fornisce un'implementazione autonoma e indipendente per la **generazione tridimensionale della superficie cutanea della mammella** (*breast skin envelope*) a partire dalla coppia standard di mammografie bi-planari 2D: vista Cranio-Caudale (**CC**) e vista Medio-Laterale Obliqua (**MLO**).

I file risultanti includono modelli standard **glTF 2.0 (`.glb`)** con materiali PBR realistici, compatibili con **Blender**, **Unity**, visori VR/AR e motori di rendering 3D.

---

## 📚 Riferimenti Scientifici e Fondamento Teorico

L'architettura del modulo si fonda su tre contributi cardine della letteratura medica e computazionale:

1. **Huang et al. (2006)** – *Reconstruction and rendering of microcalcifications from two mammogram views by modified projective grid space (MPGS)*, *Computerized Medical Imaging and Graphics*.
   - Allineamento delle viste ortogonali senza calibrazione complessa dei raggi X mediante la registrazione lungo l'asse comune del capezzolo (*Posterior Nipple Line*, PNL).
   - Costruzione della mesh superficiale continua e chiusura con la gabbia toracica del paziente.

2. **Pinto et al. (2023)** – *Generative compressed breast shape model for digital mammography and digital breast tomosynthesis*, *Medical Physics*.
   - Modellazione parametrica della sagoma mediante **super-ellissi organiche (curve di Lamé)** con esponente variabile $p(t) \in [2.0, 2.2]$ per riprodurre fedelmente la zona appiattita dei piatti di compressione e la curvatura arrotondata della cute libera.

3. **Kita, Highnam & Brady (2001)** – *Three-dimensional reconstruction of microcalcification clusters from two mammographic views*, *IEEE Transactions on Medical Imaging*.
   - Vincoli cinematici epipolari per localizzare e proiettare nello spazio 3D le formazioni biologiche e le lesioni tumorali visualizzate nelle due proiezioni.

---

## 📐 Pipeline Matematica e Geometrica

1. **Rilevamento Laterality e Pulizia Artefatti (`segmentation.py`)**:
   - Riconoscimento automatico del seno sinistro (**L**) o destro (**R**) e della parete toracica ($x = 0$ o $x = W$).
   - Soppressione delle strisce di calibrazione radiologica e dei bordi del digitalizzatore.
   - Binarizzazione con componente connessa a massima area e chiusura morfologica.
   - Individuazione dell'apice anteriore (capezzolo $(x_{apex}, y_{apex})$) e calcolo della distanza PNL:
     $$\text{PNL} = |x_{apex} - x_{chest}|$$

2. **Allineamento Spaziale e Normalizzazione Metrica (`kinematics.py`)**:
   - Parametrizzazione della profondità $t = Y / Y_{max} \in [0.0, 0.98]$.
   - Estrazione di larghezza $W(t)$ e offset $X_c(t)$ dalla vista CC.
   - Estrazione di altezza $H(t)$ e offset $Z_c(t)$ dalla vista MLO.
   - Filtraggio gaussiano $C^1$ per garantire transizioni longitudinali fluide e organiche.

3. **Lofting Super-Ellittico Continuo e Mesh Stagna (`surface_reconstruction.py`)**:
   - Per ciascuna quota $y(t)$, i punti della sezione trasversale per $\theta \in [0, 2\pi)$ sono descritti da:
     $$\begin{aligned}
     x(\theta, t) &= X_c(t) + \frac{W(t)}{2} \cdot \text{sgn}(\cos\theta) |\cos\theta|^{2/p(t)} \\
     z(\theta, t) &= Z_c(t) + \frac{H(t)}{2} \cdot \text{sgn}(\sin\theta) |\sin\theta|^{2/p(t)} \\
     y(\theta, t) &= t \cdot \text{PNL}_{mm} - \frac{x^2 + 0.6 z^2}{2 R_{chest}} \cdot (1 - t)
     \end{aligned}$$
   - con raggio toracico $R_{chest} \approx 140\text{ mm}$ ed esponente di compressione $p(t) = 2.20 - 0.20 \cdot t$.
   - Chiusura stagna posteriore sul torace e anteriore sul capezzolo, producendo una mesh **100% Manifold e Watertight**.

4. **Localizzazione Tridimensionale della Lesione (`tumor_localization.py`)**:
   - Rilevamento delle maschere ROI nelle due proiezioni per calcolare la posizione 3D $(X_{tumor}, Y_{tumor}, Z_{tumor})$.
   - Generazione di una mesh dedicata per il tumore collocata con precisione millimetrica all'interno dell'involucro cutaneo.

---

## 🚀 Istruzioni per l'Uso

### 1. Esecuzione da Riga di Comando (CLI)
```powershell
# Esecuzione per il paziente P_00001
python 06_Ricostruzione_Superficie_Seno_Paper/main.py --patient_id P_00001

# Esecuzione per un altro paziente del dataset (es. P_00005, P_00008, P_00011)
python 06_Ricostruzione_Superficie_Seno_Paper/main.py --patient_id P_00005
```

### 2. Esecuzione Rapida con Script Batch (.bat)
* **`run_reconstruction.bat`**: Doppio clic per avviare la ricostruzione guidata. Permette di inserire qualsiasi ID paziente (o premere Invio per `P_00001`). Al termine propone di aprire direttamente Blender.
* **`open_in_blender.bat`**: Doppio clic per aprire immediatamente il modello 3D generato all'interno di Blender con illuminazione studio a 3 punti e inquadratura fotocamera pronta.

---

## 📁 File Generati (`output/<PATIENT_ID>/`)

| File | Formato | Descrizione |
| :--- | :--- | :--- |
| `<PATIENT_ID>_breast_surface.glb` | **glTF Binario** | Mesh 3D stagna della superficie della pelle con materiale PBR organico. |
| `<PATIENT_ID>_breast_with_tumor.glb` | **glTF Binario** | Scena combinata con involucro cutaneo semitrasparente e tumore 3D interno evidenziato in rosso rubino. |
| `<PATIENT_ID>_breast_surface.obj` | **Wavefront OBJ** | Mesh poligonale esportata per massima compatibilità con CAD/mesh software. |
| `<PATIENT_ID>_scene.blend` | **Blender Scene** | Scena Blender completa configurata con camera ritrattistica e set luci a 3 punti. |
| `<PATIENT_ID>_preview.png` | **Immagine PNG** | Rendering fotografico ad alta risoluzione (1080p) generato automaticamente da Blender. |
| `<PATIENT_ID>_metrics.json` | **JSON Report** | Report geometrico con volume ($cm^3$), area superficiale ($cm^2$), dimensioni e verifica manifold. |
