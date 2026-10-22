# Mappa dei materiali per la tesi

Corrispondenza formale tra i capitoli del documento finale di tesi ([`thesis/2026_Tesi3___Scandone_Michele.pdf`](thesis/2026_Tesi3___Scandone_Michele.pdf)) e i moduli software, file di configurazione, manifest e dati del repository.

---

## 1. Dataset e provenienza (Capitolo 3)

- [`docs/DATASET_CREATION.md`](docs/DATASET_CREATION.md): provenienza SDSS DR17, query CasJobs, soglie Galaxy Zoo, gestione ObjID a 19 cifre, acquisizione cutout `320×320`, validazione e bilanciamento.
- [`data/manifests/metadata_summary.json`](data/manifests/metadata_summary.json): audit numerico della pulizia del catalogo CSV originale.
- [`data/manifests/manifest_summary.json`](data/manifests/manifest_summary.json): composizione statistica del dataset sperimentale bilanciato.
- [`data/manifests/validated.parquet`](data/manifests/validated.parquet): controlli di integrità, dimensioni e hash SHA-256 dei cutout.
- [`data/manifests/final.parquet`](data/manifests/final.parquet): 5.628 campioni bilanciati impiegati negli esperimenti.
- [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb): notebook di audit per l'ispezione visuale e tabellare del dataset.

**Conteggi ufficiali registrati:**
```text
65.906 righe sorgente CasJobs
65.829 candidati puliti
 6.571 cutout validi 320×320
 5.628 campioni bilanciati finali (1.876 per classe)
```

---

## 2. Partizionamento e protocollo sperimentale (Capitoli 3 e 4)

- [`docs/qtl-galaxy-classification-design.md`](docs/qtl-galaxy-classification-design.md): partizionamento a 4 fold stratificati per 10 repeat seed, definizione dei task `binary` e `three_class`, isolamento del test set e metodologia inferenziale appaiata.
- [`data/splits/all.parquet`](data/splits/all.parquet): 225.120 assegnazioni (5.628 immagini × 10 seed × 4 fold); la matrice primaria adotta `iteration=1`.
- [`configs/dataset.yaml`](configs/dataset.yaml): parametri numerici, soglie di probabilità e definizioni formali dei task.

---

## 3. Modelli e disegno sperimentale (Capitolo 4)

- [`docs/qtl-galaxy-classification-design.md`](docs/qtl-galaxy-classification-design.md): razionale scientifico del confronto controllato tra baseline classiche (B0, B1, B2) e architetture quantistiche DQC (QF, Q).
- [`configs/training.yaml`](configs/training.yaml): iperparametri di addestramento (ottimizzatore AdamW, learning rate, early stopping, regolarizzazione) per le 8 configurazioni primarie.
- [`src/tesi_qml/models/classical.py`](src/tesi_qml/models/classical.py): implementazione di LinearHead (B0), MLPHead (B1) e CapacityMatchedHead (B2-4, B2-8).
- [`src/tesi_qml/models/pennylane_qnn.py`](src/tesi_qml/models/pennylane_qnn.py): implementazione di PennylaneAngleQNN (QF, Q), PennylaneReuploadingQNN (PL_REUP) e PennylaneAmplitudeQNN (PL_AMP).
- [`src/tesi_qml/models/circuit_spec.py`](src/tesi_qml/models/circuit_spec.py): specifica formale del circuito quantistico variazionale.
- [`scripts/train.py`](scripts/train.py): pipeline di addestramento, salvataggio checkpoint atomico e calcolo metriche.

---

## 4. Matrice primaria e campagna subset (Capitoli 5 e 6)

- [`src/tesi_qml/matrix.py`](src/tesi_qml/matrix.py) e [`scripts/run_matrix.py`](scripts/run_matrix.py): runner locale riprendibile per i 160 esperimenti della matrice primaria full.
- [`scripts/colab_session.py`](scripts/colab_session.py): orchestratore per l'esecuzione della matrice full su Google Colab con persistenza atomica su Google Drive.
- [`scripts/colab_subsets.py`](scripts/colab_subsets.py): orchestratore della campagna secondaria sui subset (100, 300, 900 campioni, 480 run complessivi).
- [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb): notebook Colab unificato per il ciclo operativo completo (matrice full, subset e aggregazione finale).
- [`COLAB_GUIDE.md`](COLAB_GUIDE.md): guida operativa per la configurazione e ripresa delle sessioni su Google Colab.

---

## 5. Risultati, inferenza statistica e figure (Capitoli 5 e 6)

- [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md): specifica formale degli artefatti prodotti da ogni singolo run e delle tabelle aggregate.
- [`scripts/aggregate_results.py`](scripts/aggregate_results.py): calcolo delle metriche aggregate, intervalli di confidenza bootstrap BCa al 95%, permutation test con sign-flip bilaterale, correzione Holm-Bonferroni e Cohen's $d_z$.
- [`scripts/generate_figures.py`](scripts/generate_figures.py): generazione deterministica di tutte le figure vettoriali in formato PDF.
- [`figures/`](figures/): directory contenente le figure vettoriali prodotte e incluse nel testo della tesi.
- [`notebooks/02_experiment_analysis.ipynb`](notebooks/02_experiment_analysis.ipynb): analisi quantitativa e tabelle dei risultati aggregati.
- [`notebooks/03_thesis_figures.ipynb`](notebooks/03_thesis_figures.ipynb): ispezione ed esportazione dei grafici della tesi.

---

## 6. Interpretabilità Visiva XAI (Capitolo 6)

- [`heat_map/gradcam_analysis.py`](heat_map/gradcam_analysis.py): modulo per l'estrazione e sovrapposizione di mappe Grad-CAM su layer4 di ResNet-18 per modelli B2, QF e Q.
- [`heat_map/07_colab_gradcam.ipynb`](heat_map/07_colab_gradcam.ipynb): notebook autonomo per l'esecuzione del protocollo Grad-CAM in ambiente Colab/GPU.
- [`heat_map/test_heatmaps.py`](heat_map/test_heatmaps.py): test suite per la validazione delle ricostruzioni di probabilità e mappe di salienza.
- [`heat_map/README.md`](heat_map/README.md): documentazione metodologica del modulo Grad-CAM.

---

## 7. Tracciabilità e limiti metodologici (Capitolo 7)

- [`src/tesi_qml/reproducibility.py`](src/tesi_qml/reproducibility.py): determinismo, gestione seed, fingerprint di configurazione e registrazione ambiente software.
- [`data/manifests/rejected.parquet`](data/manifests/rejected.parquet): record esclusi durante la fase di pulizia iniziale.
- [`data/manifests/manifest_exclusions.parquet`](data/manifests/manifest_exclusions.parquet): immagini valide conservate come riserva durante il bilanciamento.
- **Nota sui filtri fotometrici SDSS DR17**: il file CSV esportato da CasJobs non conserva le colonne `clean`, `type` e `g`. I relativi criteri di selezione sono definiti a monte nella query CasJobs ufficiale e garantiti dall'estrazione originale.
