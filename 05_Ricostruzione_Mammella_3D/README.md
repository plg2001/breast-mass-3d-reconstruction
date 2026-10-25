# Modulo 05: Ricostruzione 3D della Superficie della Mammella (Skin Envelope)

Questo modulo implementa la ricostruzione tridimensionale della **superficie cutanea esterna della mammella** a partire da una coppia di mammografie bi-planari (viste Cranio-Caudale **CC** e Medio-Laterale Obliqua **MLO**), in formato JPEG o DICOM.

Il modello ignora l'interno biologico (dotti, grasso, ecc.) e genera un guscio anatomico continuo, chiuso (*watertight*) e metricamente calibrato, ottimizzato per:
1. **L'applicazione immersiva su visore Android / Unity** (descritta nella proposta di tesi).
2. **L'integrazione spaziale con la lesione 3D** (ricostruita dai moduli Ling Shao o SMRD).
3. **La visualizzazione simultanea delle radiografie 2D originali** all'interno della mesh semitrasparente.

---

## 📐 Fondamento Matematico e Geometrico

La pipeline si basa sui principi di cinematica mammografica e geometria multiview (Huang et al. 2006, Rodríguez-Ruiz 2017, Pinto et al. 2022):

1. **Segmentazione Bordo Pelle (*Skin Line*)**:
   - Pulizia automatica delle cornici di scansione radiologica.
   - Binarizzazione adattiva e selezione della componente connessa a massima area.
   - Rilevamento automatico della laterality (destra/sinistra), parete toracica (*chest wall*) e apice anteriore (*capezzolo*).

2. **Allineamento Spaziale lungo l'Asse PNL**:
   - L'asse comune capezzolo-torace (*Posterior Nipple Line*, asse $Y$) normalizza le due viste ortogonali.
   - Vista CC: definisce la larghezza $W(y)$ e l'offset mediale-laterale $X_c(y)$.
   - Vista MLO: definisce l'altezza $H(y)$ e l'offset supero-inferiore $Z_c(y)$.

3. **Lofting Quad-Ellittico Continuo ($C^1$ Smooth)**:
   - Ogni sezione trasversale ad una profondità $y \in [0, Y_{nipple}]$ viene modellata come una super-ellisse continua che tocca esattamente i 4 margini radiologici reali:
     $$\begin{aligned}
     x(\theta, y) &= X_c(y) + \frac{W(y)}{2} \cdot \text{sgn}(\cos\theta) |\cos\theta|^{2/p} \\
     z(\theta, y) &= Z_c(y) + \frac{H(y)}{2} \cdot \text{sgn}(\sin\theta) |\sin\theta|^{2/p}
     \end{aligned}$$
   - con esponente organico $p \approx 2.1$.

4. **Raccordo Toracico e Chiusura Watertight**:
   - Alla base posteriore ($y = 0$), una curvatura convessa simula il raccordo naturale con la cassa toracica.
   - All'apice anteriore ($y = Y_{nipple}$), la mesh converge dolcemente sul capezzolo.
   - Un filtro di smoothing laplaciano conservativo garantisce la finitura organica della cute.

---

## 🚀 Avvio Rapido

### Da Riga di Comando:
```powershell
# Esecuzione per il paziente P_00005 (default)
python 05_Ricostruzione_Mammella_3D/breast_surface_reconstruction.py --patient_id P_00005

# Esecuzione per un paziente specifico del dataset
python 05_Ricostruzione_Mammella_3D/breast_surface_reconstruction.py --patient_id P_00008

# Esecuzione con percorsi file personalizzati
python 05_Ricostruzione_Mammella_3D/breast_surface_reconstruction.py --cc_path data/P_00005/R-CC.jpg --mlo_path data/P_00005/R-MLO.jpg
```

### Tramite Script Batch:
* Fai doppio clic su `run_surface_reconstruction.bat` per eseguire la ricostruzione automatica e visualizzare la figura riassuntiva.
* Fai doppio clic su `open_in_blender.bat` per aprire il modello 3D generato nel visualizzatore 3D / Blender.

---

## 📂 File Generati per Ciascun Paziente (`output/<PATIENT_ID>/`)

| File | Descrizione |
| :--- | :--- |
| `<PATIENT_ID>_breast_surface_multiview.png` | **Figura clinica a 6 pannelli** ad alta risoluzione (Vista laterale MLO, Vista frontale CC, Vista superiore con assi, Mesh wireframe 3D, Sezioni 2D trasparenti intersecanti, Vista posteriore con scala metrica). |
| `breast_skin_surface.obj` | File Wavefront OBJ contenente solo la mesh della superficie cutanea (compatibile Unity, Blender, 3ds Max). |
| `breast_skin_surface.glb` | File binario glTF con materiale PBR per la pelle (leggero, ottimizzato per visore VR / mobile). |
| `breast_skin_with_mammogram_slices.glb` | Scena 3D completa contenente la superficie semitrasparente e le due lastre mammografiche collocate sui rispettivi piani nello spazio 3D. |
| `surface_metrics.json` | Report geometrico con volume ($cm^3$), area ($cm^2$), dimensioni bounding box ($mm$), numero di poligoni e verifica di watertightness. |
