# Modulo 07: Ricostruzione 3D Totale (Mammella a 14 Strati + Lesione Tumorale SMRD)

Questo modulo rappresenta il vertice integrativo del progetto di tesi: unisce la bio-modellazione anatomica multi-strato della mammella con la ricostruzione matematica avanzata della lesione tumorale spicolata, localizzandola nello spazio tridimensionale mediante geometria epipolare bi-planare.

---

## 🔬 Fondamento Scientifico e Integrazione dei Paper

Il modulo combina e armonizza 5 pilastri della letteratura scientifica internazionale:

1. **Luis de Sisternes et al. (2015)** – *A computational model to generate simulated three-dimensional breast masses*, *Medical Physics*, 42(2), 1098-1118 (PMC4320152).
   * **Modello SMRD**: Deformazione radiale su armoniche sferiche reali ($l_{\max} = 6$) regolarizzate con operatore di Laplace-Beltrami.
   * **Spicule Engine 3D**: Estrazione delle spicole 2D dalle radiografie e sintesi di un pattern spicolato a raggiera 3D (carcinomi infiltranti).

2. **Yasuyo Kita, Ralph Highnam, and Michael Brady (2001)** – *Correspondence between different views of breast X-rays using curved epipolar lines*, *IEEE Transactions on Medical Imaging / Medical Image Analysis*.
   * Calcolo della linea epipolare curva e triangolazione stereotassica del centroide tumorale combinando la vista Cranio-Caudale (CC) e Medio-Laterale Obliqua (MLO).

3. **Jin-shan Huang et al. (2006)** – *Reconstruction and rendering of microcalcifications from two mammogram views by modified projective grid space (MPGS)*, *Computerized Medical Imaging and Graphics*.
   * Normalizzazione della cinematica lungo l'asse della linea capezzolo-pettorale (*Posterior Nipple Line*, PNL) e determinazione univoca della profondità antero-posteriore $Z$.

4. **Franklyn Pinto et al. (2023)** – *Generative compressed breast shape model for digital mammography and digital breast tomosynthesis*, *Medical Physics*.
   * Involucro cutaneo organico a super-ellissi (curve di Lamé) e vincolo di confinamento fisiologico del tumore all'interno del mantello parenchimale.

5. **FDA VICTRE (Graff et al. 2016, IEEE TMI) + 3D U-Net-GAN (2026)**:
   * Generazione dei 14 strati biologici completi: derma, capezzolo, vene, grasso premammario solido, nervi cutanei, dotti galattofori con acini TDLU, legamenti di Cooper, vasi linfatici, linfonodi, arterie, cuscino retromammario, fascia pettorale e muscolo grande pettorale.

---

## 📐 Struttura della Scena 3D (15 Oggetti PBR in Blender)

| # | Nome Oggetto Blender | Struttura Biologica / Patologica | Proprietà Materiale PBR |
|---|---|---|---|
| 1 | `breast_skin` | Cute / Derma esterno continuo | Rosa carne tenue, Alpha 0.22, SSS |
| 2 | `nipple_areola` | Complesso Areola-Capezzolo (NAC) | Pigmento melanico naturale, Alpha 0.98 |
| 3 | `breast_veins` | Plesso venoso superficiale di Haller | Blu venoso reale, Alpha 0.92 |
| 4 | `subcutaneous_fat` | Mantello adiposo volumetrico compatto | Avorio caldo corposo, Alpha 0.52, SSS |
| 5 | `mammary_nerves` | Nervi cutanei sensoriali (4° intercostale) | Giallo brillante anatomico, Alpha 0.98 |
| 6 | `lymphatic_vessels` | Plesso di Sappey e vasi ascellari | Verde-oro luminescente, Alpha 0.92 |
| 7 | `lymph_nodes` | Linfonodi intramammari reniformi | Verde oliva chiaro / buff, Alpha 0.90 |
| 8 | `cooper_ligaments` | Legamenti sospensori di Cooper | Cordoni fibrosi argentei, Alpha 0.85 |
| 9 | `fibroglandular_tissue`| Parenchima rugoso U-Net-GAN | Arancio ambrato, Alpha 0.42 |
| 10| `lactiferous_ducts` | Albero galattoforo e acini TDLU | Bianco latte perlaceo opaco, Alpha 0.94 |
| 11| `breast_arteries` | Rete arteriosa ghiandolare profonda | Rosso scarlatto arterioso, Alpha 0.95 |
| 12| `retromammary_fat` | Cuscino retromammario (Chassaignac) | Giallo avorio solido, Alpha 0.48 |
| 13| `deep_pectoral_fascia`| Fascia connettivale pre-pettorale | Velo fibroso perlaceo, Alpha 0.38 |
| 14| `pectoral_muscle` | Muscolo grande pettorale toracico | Rosso muscolare profondo, Alpha 0.85 |
| 15| **`tumor_lesion`** | **Massa Tumorale Spicolata SMRD** | **Porpora ematico scuro, Alpha 0.98, SSS, spicole riflettenti** |

---

## 🚀 Avvio Rapido

### Con Script Batch (.bat)
* **`run_total_reconstruction.bat`**: Doppio clic per avviare la pipeline guidata (premi Invio per `P_00081` o inserisci `P_00001`, `P_00005`, ecc.).
* **`open_in_blender.bat`**: Doppio clic per aprire direttamente la scena 3D fotorealistica in Blender.

### Da Terminale (CLI)
```powershell
python 07_Total_Reconstruction_Module/src/total_reconstruction.py --patient_id P_00081
```

---

## 📁 File Generati (`output/<PATIENT_ID>/`)
* `<PID>_total_3d.glb`: Modello 3D unificato per visori VR, web viewer o Unity.
* `<PID>_total_3d.blend`: Scena Blender con luci studio a 3 punti e shader PBR calibrati.
* `<PID>_total_preview.png`: Rendering 3D a 3/4 ad alta risoluzione.
* `<PID>_total_preview_profile.png`: Rendering diagnostico del profilo anatomico laterale (MLO).
* `<PID>_total_clinical_report.json`: Report clinico metrologico con coordinate $(X, Y, Z)$, quadrante anatomico (es. UOQ), volume tumorale in $\text{mm}^3$ e conta delle spicole.
* `breast_3d_volume.npy`: Matrice volumetrica neurale tridimensionale U-Net-GAN.
