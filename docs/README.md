# Documentazione Metodologica e Tecnica

Questa cartella raccoglie le specifiche formali, i protocolli metodologici e i documenti di design scientifico alla base del progetto di tesi:
**"Classificazione ibrida classico-quantistica delle morfologie galattiche: confronto tra bottleneck classici e circuiti quantistici variazionali"**.

---

## 📚 Indice dei Documenti

| Documento | Oggetto e Contenuto |
|---|---|
| [`DATASET_CREATION.md`](DATASET_CREATION.md) | **Provenienza e preparazione dati**: query SQL CasJobs su SDSS DR17, criteri di selezione Galaxy Zoo, filtraggio flag `clean`, risoluzione e validazione cutout `320×320`, bilanciamento a 5.628 campioni ($1.876$ per classe). |
| [`qtl-galaxy-classification-design.md`](qtl-galaxy-classification-design.md) | **Disegno sperimentale formale**: razionale scientifico del confronto tra architetture DQC PennyLane e baseline classiche a parità di gradi di libertà ($B0$, $B1$, $B2$, $QF$, $Q$), schema di partizionamento a 4 fold per 10 repeat seed e matrice primaria da 160 run. |
| [`OUTPUT_SCHEMA.md`](OUTPUT_SCHEMA.md) | **Schema degli artefatti sperimentali**: formato atomico di `metrics.json`, `run_config.json`, `history.jsonl`, `test_predictions.parquet`, registri di matrice e tabelle aggregate sia per la matrice primaria sia per la campagna subset. |
| [`COLAB_EXECUTION_PROCESS.md`](COLAB_EXECUTION_PROCESS.md) | **Procedura operativa su Google Colab Pro**: gestione della quota GPU, backup incrementale su Google Drive, archiviazione atomica degli stati (`colab_state.tar.gz`, `colab_subset_state.tar.gz`), matrice primaria (160 run) e campagna subset (480 run). |

---

## 🏛️ Tracciabilità e Riproducibilità

Tutti i parametri numerici, le soglie di probabilità e le definizioni delle partizioni documentati in questa cartella sono bloccati nei file di configurazione formali:
- [`configs/dataset.yaml`](../configs/dataset.yaml)
- [`configs/training.yaml`](../configs/training.yaml)

Per la corrispondenza dettagliata tra capitoli della tesi e documenti tecnici, fare riferimento alla [Mappa dei materiali per la tesi](../THESIS_FILE_MAP.md).
