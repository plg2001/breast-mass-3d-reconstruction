# SMRD: Spherical-Harmonic Multi-view Radial Deformation
## Ricostruzione 3D Avanzata di Lesioni Mammarie con Spicule Engine (CC + MLO)

Questo modulo implementa lo stato dell'arte per la ricostruzione 3D di masse tumorali mammarie da proiezioni mammografiche ortogonali/oblique, basato sul lavoro di riferimento di **Luis de Sisternes et al. (Medical Physics, 2015)** integrato con i vincoli stereotassici epipolari di **Sheng-Chih Yang et al. (2007)** e **Nico Karssemeijer et al.**.

---

## 1. Fondamento Scientifico di Riferimento

* **Articolo Cardine:**  
  *Luis de Sisternes, Jovan G. Brankov, Adam M. Zysk, Robert A. Schmidt, Robert M. Nishikawa, Miles N. Wernick (2015).*  
  **"A computational model to generate simulated three-dimensional breast masses."**  
  *Medical Physics*, 42(2), 1098–1118. [DOI: 10.1118/1.4905232 | PMCID: PMC4320152].
  * **Sezione II.A:** Modello della massa centrale in armoniche sferiche / GRS.
  * **Sezione II.B:** Modello di spicolatura 3D (*Spicules & Fractal Branching*) con aghi radiali conici acuti estrusi nel parenchima.

* **Vincoli Multi-View Stereotassici:**  
  *Sheng-Chih Yang et al. (2007)* — Normalizzazione epipolare e triangolazione riferita al capezzolo tra proiezioni CC (0°) e MLO (45°).

---

## 2. Architettura a Due Stadi (De Sisternes 2015)

1. **Stadio 1 (Massa Centrale Organica - *Central Mass*):**  
   Il corpo tumorale primario viene ricostruito come una varietà 2-manifold chiusa $R_{\text{base}}(\theta, \phi)$ tramite espansione in **armoniche sferiche reali** regolarizzata dall'energia di curvatura di Laplace-Beltrami. Questo stadio garantisce continuità differenziabile, eliminando gli spigoli a 45° del vecchio metodo Ling Shao.

2. **Stadio 2 (Motore di Spicolatura 3D - *Spicule Engine*):**  
   L'algoritmo rileva le vere "spikes" (creste acuminate) lungo i contorni ad alta frequenza delle mammografie CC ed MLO. Queste spikes vengono quindi proiettate radialmente nello spazio tridimensionale come aghi conici affilati:
   $$\Delta R_s(\theta, \phi) = H_s \cdot \exp\left( - \frac{\psi(\hat{\mathbf{u}}(\theta, \phi), \hat{\mathbf{d}}_s)^2}{2 \sigma_s^2} \right)$$
   con larghezza angolare $\sigma_s \approx 0.055\text{ rad}$ (~3.2°), donando al solido il caratteristico aspetto stellato a raggiera del carcinoma infiltrante spicolato.

---

## 3. Risultati Sperimentali

### Paziente `P_00001` (Paziente 1)
* **Volume:** $14273.05\text{ mm}^3$ ($14.27\text{ cm}^3$)
* **Sfericità:** $\Psi = 0.724$ (tipica di carcinoma spicolato infiltrante)
* **Spikes 3D Radianti:** **89 spine** acuminate (Spike massima: **4.42 mm**)
* **Topologia:** Watertight Manifold chiusa (Eulero = 2, 28.802 vertici, 57.600 triangoli)

### Paziente `P_00232` (Caso Benchmark 1)
* **Volume:** $3101.11\text{ mm}^3$ ($3.10\text{ cm}^3$)
* **Sfericità:** $\Psi = 0.884$ (massa prevalentemente circoscritta/nodulare)
* **Spikes 3D Radianti:** **75 spine** (Spike massima: **0.97 mm**)

### Paziente `P_01226` (Caso con Frastagliature e Spikes Evidenti)
* **Volume:** $10668.62\text{ mm}^3$ ($10.67\text{ cm}^3$)
* **Sfericità:** $\Psi = 0.849$
* **Spikes 3D Radianti:** **66 spine** (Spike massima: **2.08 mm**)

---

## 4. Come Utilizzare Questo Modulo

* **Ricostruzione Paziente 1 con Spikes:**
  ```powershell
  python smrd_lesion_reconstruction.py --patient P_00001 --spicularity 1.2
  ```
* **Visualizzazione Immediata in Blender:**
  Fai doppio clic su `open_in_blender.bat` e seleziona `1` (per `P_00001`) oppure `2` (per `P_00232`) o `3` (per `P_01226`).
* **Regolazione Intensità Spikes:**
  Il parametro `--spicularity` controlla quanto sono pronunciate le spine (default 1.2, impostabile da 0.0 per superficie liscia fino a 2.0 per spine molto marcate).
