# Schema degli output

## Dataset e split

- `data/manifests/final.parquet`: manifest deduplicato e bilanciato con `objid`, `label`, `class_index`, path, SHA-256 e metadata scientifici.
- `data/manifests/manifest_exclusions.parquet`: esclusioni persistite e relativa motivazione.
- `data/splits/all.parquet`: sorgente triclasse con `repeat_seed`, `iteration`, `base_fold`, `split`, `objid`, `label`, `class_index`.

Il binario è una vista filtrata della tabella triclasse; gli objid elliptical/spiral conservano lo stesso split.

## Feature bank

Path: `data/features/seed_<seed>/iteration_<iteration>/<split>/`.

- `features.pt`: matrice float32 `[campioni×viste, 512]`;
- `metadata.parquet`: `objid`, label/nome label, view, view seed, indice feature e provenance per campione;
- `provenance.json`: SHA-256 manifest, backbone, trasformazione, repeat/iteration/split, viste e fingerprint.

Train usa una vista; validation e test sono deterministici. La provenance task-specific viene aggiunta alla vista filtrata in memoria.

## Matrice

`results/matrix/primary.json` contiene 160 righe con `run_id`, task, modello, q, seed, iterazione, stato, path ed eventuale errore. Gli stati ammessi sono `pending`, `running`, `complete`, `failed`.

## Run

Path:

```text
results/runs/<task>/<model_tag>/seed_<seed>/iteration_1/config_<fingerprint>/
```

Artefatti obbligatori:

- `run_config.json`: task/mapping, modello, q/depth/freeze, repeat/sub-seed, configurazioni, provenance feature, checksum iniziali, fingerprint, ambiente e device;
- `checkpoints/best.pt`: minimo validation loss;
- `checkpoints/last.pt`: ultimo stato riprendibile;
- `history.jsonl`: metriche e gradienti per epoca;
- `metrics.json`, `predictions.parquet`: validation sul checkpoint best;
- `test_metrics.json`, `test_predictions.parquet`: test sigillato, aperto dopo il restore del best;
- `figures/*.png|pdf`: figure derivate.

`history.jsonl` include almeno loss/accuracy/F1 train-validation, learning rate, tempo, norma dei gradienti e, per Q, norma/varianza dei gradienti quantistici.

`test_metrics.json` include accuracy, balanced accuracy, precision/recall/F1 micro-macro-weighted e per classe, confusion matrix raw/normalizzata, ROC-AUC OvR, PR-AUC, log loss, Brier, ECE, runtime, epoca best, memoria e conteggi di parametri. QF include il checksum finale dei pesi congelati.

`test_predictions.parquet` include `objid`, classe vera/predetta, correttezza, confidence, entropy, logits/probabilità dinamiche per classe, task, modello, q, freeze, seed/fold e fingerprint.

## Aggregati

`results/aggregate/` contiene:

- `runs.parquet`: una riga per run completo;
- `summary.parquet|json`: statistiche per task, modello e q usando il repeat seed come unità inferenziale;
- `binary.parquet`, `three_class.parquet`: tabelle task-specific;
- `merger.parquet`: precision, recall e F1 dei merger;
- `primary_hypotheses.parquet|json`: quattro Q−B2 con differenza media/deviazione standard, bootstrap CI, sign-flip p, Holm p, Cohen dz e Wilcoxon p.

Run parziali, duplicati, extra o con fingerprint incompatibili non entrano nell’aggregazione primaria.

## Stato Colab

`colab_state.tar.gz` è l’archivio atomico ripristinato a ogni sessione. Contiene
`results/` e `data/features/`, esclude i file temporanei `.part` e rifiuta membri
TAR che tentano di scrivere fuori dalla project root. Gli archivi iniziali
`colab_code.tar.gz` e `colab_dataset.tar.gz` restano immutabili su Drive.
