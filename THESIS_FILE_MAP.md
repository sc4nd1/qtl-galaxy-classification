# Mappa dei materiali per la tesi

Questa guida indica quali file usare come fonte per ciascuna parte della tesi. I
Markdown sono bozze tecniche: prima della consegna vanno integrati con citazioni
bibliografiche e adattati allo stile dell'ateneo.

## Dataset e provenienza

- `dataset_creation.md`: SDSS DR17, query CasJobs, soglie Galaxy Zoo, ObjID,
  acquisizione `320×320`, validazione e bilanciamento.
- `data/manifests/metadata_summary.json`: evidenza numerica della pulizia del CSV.
- `data/manifests/manifest_summary.json`: evidenza numerica del dataset finale.
- `data/manifests/validated.parquet`: controlli, dimensioni e hash dei cutout.
- `data/manifests/final.parquet`: 5.628 campioni effettivamente usati.

Conteggi da mantenere coerenti nel testo:

```text
65.906 righe sorgente
65.829 candidati puliti
6.571 cutout validi 320×320
5.628 campioni finali
1.876 campioni per classe
```

## Partizionamento e protocollo

- `dataset_methodology_k4.md`: quattro fold, 10 repeat seed, task binario e
  triclasse, protocollo primario e inferenza statistica.
- `data/splits/all.parquet`: 225.120 assegnazioni, corrispondenti a 10 seed e 4
  iterazioni; la matrice primaria usa soltanto `iteration=1`.
- `configs/dataset.yaml`: soglie, seed e definizione dei task.

## Disegno sperimentale

- `2026-08-27-qtl-galaxy-classification-design.md`: razionale completo delle
  baseline e della matrice da 160 run.
- `configs/training.yaml`: iperparametri bloccati e otto configurazioni primarie.
- `src/tesi_qml/experiments.py` e `src/tesi_qml/matrix.py`: identità dei run e
  materializzazione della matrice.

## Implementazione PennyLane

- `docs/PENNYLANE_IMPLEMENTATIONS.md`: varianti circuitali e scelte implementative.
- `src/tesi_qml/models/circuit_spec.py`: specifica del circuito.
- `src/tesi_qml/models/pennylane_qnn.py`: implementazione Q e QF.
- `src/tesi_qml/models/classical.py`: baseline B0, B1 e B2.
- `scripts/train.py`: flusso di addestramento e valutazione.

## Risultati, statistiche e figure

- `docs/OUTPUT_SCHEMA.md`: formato degli artefatti prodotti da ogni run.
- `scripts/aggregate_results.py`: confronti appaiati, intervalli e correzione Holm.
- `scripts/generate_figures.py`: figure finali.
- `notebooks/02_experiment_analysis.ipynb`: ispezione dei risultati aggregati.
- `notebooks/03_thesis_figures.ipynb`: esportazione di grafici per la tesi.

## Riproducibilità e limiti

- `src/tesi_qml/reproducibility.py`: seed, fingerprint e ambiente software.
- `data/manifests/rejected.parquet`: record esclusi durante la pulizia.
- `data/manifests/manifest_exclusions.parquet`: immagini valide tenute come
  riserva per il bilanciamento.
- `COLAB_GUIDE.md`: procedura operativa e ripresa delle sessioni.

Nel capitolo dei limiti va dichiarato che il CSV esportato non conserva le colonne
`clean`, `type` e `g`: tali filtri sono documentati dalla query CasJobs, ma non
sono riauditabili usando soltanto il CSV locale.
