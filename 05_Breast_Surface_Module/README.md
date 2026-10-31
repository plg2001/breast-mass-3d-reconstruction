# Modulo 05: Ricostruzione 3D della Superficie del Seno da Mammografie 2D (Paper Huang / Pinto / Kita)

Questo modulo fornisce un'implementazione autonoma per la **generazione tridimensionale della superficie cutanea pura della mammella** (*breast skin envelope*) a partire dalla coppia standard di mammografie bi-planari 2D: vista Cranio-Caudale (**CC**) e vista Medio-Laterale Obliqua (**MLO**).

I file risultanti includono modelli standard **glTF 2.0 (`.glb`)** con materiale PBR opaco realistico per la pelle, compatibili con **Blender**, **Unity**, visori VR/AR e motori di rendering 3D.

---

## 📚 Riferimenti Scientifici e Fondamento Teorico

1. **Huang et al. (2006)** – *Reconstruction and rendering of microcalcifications from two mammogram views by modified projective grid space (MPGS)*, *Computerized Medical Imaging and Graphics*.
   - Allineamento delle viste ortogonali senza calibrazione complessa dei raggi X mediante la registrazione lungo l'asse comune del capezzolo (*Posterior Nipple Line*, PNL).
   - Costruzione della mesh superficiale continua e chiusura con la parete toracica della paziente.

2. **Pinto et al. (2023)** – *Generative compressed breast shape model for digital mammography and digital breast tomosynthesis*, *Medical Physics*.
   - Modellazione parametrica della sagoma mediante **super-ellissi organiche (curve di Lamé)** con esponente variabile $p(t) \in [2.0, 2.15]$ per riprodurre fedelmente la zona appiattita dei piatti di compressione e la curvatura arrotondata della cute libera.

3. **Rodríguez-Ruiz et al. (2017)** – *The compressed breast during mammography and breast tomosynthesis: in vivo shape characterization and modeling*, *Physics in Medicine & Biology*.
   - Caratterizzazione in vivo della curvatura radiale della superficie cutanea e raccordo toracico.

---

## 📐 Orientamento Anatomico Verticale Eretto

Il modello 3D è generato in **posizione eretta anatomica** standard:
* **Asse $Z$ (Verticale)**: Altezza anatomica del seno (polo superiore verso l'alto $+Z$, solco sottomammario verso il basso $-Z$).
* **Asse $X$ (Orizzontale)**: Larghezza medio-laterale da vista CC.
* **Asse $Y$ (Profondità)**: Proiezione orizzontale dal piano toracico ($Y = 0$) in avanti verso il capezzolo.

---

## 🚀 Istruzioni per l'Uso

### 1. Esecuzione con Script Batch (.bat)
* **`run_reconstruction.bat`**: Doppio clic per avviare la ricostruzione guidata. Inserisci l'ID del paziente (es. premi Invio per `P_00081`, oppure digita `P_00001`, `P_00005`, ecc.). Al termine propone di aprire direttamente Blender.
* **`open_in_blender.bat`**: Doppio clic per aprire istantaneamente il modello 3D generato in Blender con set luci studio a 3 punti e telecamera già inquadrata.

### 2. Esecuzione da Riga di Comando (CLI)
```powershell
# Esecuzione per il paziente P_00081
python 05_Breast_Surface_Module/main.py --patient_id P_00081

# Esecuzione per un altro paziente del dataset (es. P_00001, P_00005)
python 05_Breast_Surface_Module/main.py --patient_id P_00001
```

---

## 📁 File Generati (`output/<PATIENT_ID>/`)

| File | Formato | Descrizione |
| :--- | :--- | :--- |
| `<PATIENT_ID>_breast_surface.glb` | **glTF Binario** | Mesh 3D stagna (*watertight*) della sola superficie cutanea con materiale PBR opaco naturale per la pelle. |
| `<PATIENT_ID>_breast_surface.obj` | **Wavefront OBJ** | Mesh poligonale standard compatibile con tutti i software CAD / 3D. |
| `<PATIENT_ID>_scene.blend` | **Blender Scene** | Scena Blender completa configurata con telecamera prospettica 3/4 e set luci studio a 3 punti. |
| `<PATIENT_ID>_preview.png` | **Immagine PNG** | Rendering fotografico 1080p generato automaticamente da Blender. |
| `<PATIENT_ID>_metrics.json` | **JSON Report** | Report geometrico con volume ($cm^3$), area superficiale ($cm^2$), dimensioni e verifica manifold. |
