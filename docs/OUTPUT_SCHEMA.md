# Schema degli output

## 1. Dataset e split

- `data/manifests/final.parquet`: manifest deduplicato e bilanciato con `objid`, `label`, `class_index`, path, SHA-256 e metadati astrofisici.
- `data/manifests/manifest_exclusions.parquet`: esclusioni tracciate e relativa motivazione scientifica.
- `data/splits/all.parquet`: tabella unificata con `repeat_seed`, `iteration`, `base_fold`, `split`, `objid`, `label`, `class_index` (225.120 righe).

Il task `binary` è una vista filtrata della tabella triclasse; gli ObjID `elliptical` e `spiral` conservano rigorosamente la medesima assegnazione di fold.

---

## 2. Feature bank

Percorso: `data/features/seed_<seed>/iteration_<iteration>/<split>/`

- `features.pt`: tensore PyTorch `float32` di dimensione `[campioni, 512]`;
- `metadata.parquet`: `objid`, etichette, visualizzazioni, indice e provenance per ciascun campione;
- `provenance.json`: hash SHA-256 dei manifest, identificativo backbone, trasformazioni applicate, repeat seed, iterazione e fingerprint.

---

## 3. Registri della matrice sperimentale

- `results/matrix/primary.json`: registro dei 160 run della matrice primaria full. Contiene per ogni esperimento: `run_id`, `task`, `model`, `q`, `seed`, `iteration`, `status` (`pending`, `running`, `complete`, `failed`), percorso cartella ed eventuale traccia di errore.
- `results/matrix/subset_<size>.json` (con `size` $\in \{100, 300, 900\}$): registri da 160 run per ciascuna dimensione della campagna secondaria (480 run complessivi).

---

## 4. Struttura del singolo run

Percorso standard:
```text
results/runs/<task>/<model_tag>/seed_<seed>/iteration_1/config_<fingerprint>/
```

Artefatti obbligatori per la validazione di completezza:
- `run_config.json`: specifica completa di task, modello, qubit ($q$), profondità ($L$), freeze, repeat seed, fingerprint, ambiente e hardware;
- `checkpoints/best.pt`: checkpoint dei pesi corrispondente al minimo della validation loss;
- `checkpoints/last.pt`: ultimo checkpoint di stato per la ripresa deterministica;
- `history.jsonl`: traccia sequenziale di loss, accuracy, F1, learning rate, tempi per epoca e norma dei gradienti (inclusi gradienti quantistici per $Q$);
- `metrics.json`, `predictions.parquet`: metriche e inferenze sul validation set al checkpoint ottimale;
- `test_metrics.json`, `test_predictions.parquet`: metriche ed inferenze sul test set sigillato, valutato esclusivamente dopo il ripristino di `best.pt`. Per $QF$ include il checksum dei pesi quantistici congelati;
- `figures/*.png|pdf`: curve di apprendimento e matrici di confusione generate per il singolo run.

---

## 5. Tabelle e statistiche aggregate

Percorso: `results/aggregate/`

### Matrice primaria full (Capitolo 5)
- `runs.parquet`: tabella consolidata dei 160 run primari completati;
- `summary.parquet` e `summary.json`: metriche aggregate (media, deviazione standard, mediana, IQR) per task, modello e numero di qubit con unità inferenziale sul repeat seed ($n=10$);
- `binary.parquet`: aggregato specifico per la classificazione binaria;
- `three_class.parquet`: aggregato specifico per la classificazione triclasse;
- `merger.parquet`: metriche dettagliate per la classe morfologica merger (precision, recall, F1);
- `primary_hypotheses.parquet` e `primary_hypotheses.json`: risultati dei 4 test di ipotesi primari $Q - B2$ con differenza media, deviazione standard, intervallo bootstrap BCa al 95%, p-value da permutation test con sign-flip bilaterale, p-value corretto Holm-Bonferroni, Cohen's $d_z$ e test di Wilcoxon.

### Campagna secondaria subset (Capitolo 6)
- `runs_with_training_size.parquet`: tabella unificata di tutti i 640 run (160 full + 480 subset);
- `subset_learning_curve.parquet` e `subset_learning_curve.json`: curve di apprendimento della macro-F1 in funzione dei campioni di training ($100, 300, 900, \text{full}$);
- `secondary_subset_hypotheses.parquet` e `secondary_subset_hypotheses.json`: test di permutazione Holm-Bonferroni per i confronti secondari $Q - B2$ condotti sui singoli sottoinsiemi.

---

## 6. Archivi persistenti Google Drive

- `colab_state.tar.gz`: archivio atomico della matrice primaria full. Contiene `results/` e `data/features/`.
- `colab_subset_state.tar.gz`: archivio atomico della campagna secondaria sui subset.
- Gli archivi escludono i file temporanei con estensione `.part` e verificano che i percorsi interni non risalgano all'esterno della radice di progetto.
