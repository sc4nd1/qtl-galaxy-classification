# Guida operativa per Google Colab

Il pacchetto `qtl-galaxy-classification` contiene codice, manifest, split e
documentazione, ma non contiene immagini. Prima di creare gli archivi Colab copia
i 6.571 cutout `320×320` nelle tre cartelle:

```text
data/images/elliptical/   2.347 file
data/images/merger/       1.877 file
data/images/spiral/       2.347 file
```

I nomi devono rimanere gli ObjID esatti presenti in `downloads.parquet`.

## 1. Controllo locale

Da PowerShell, nella radice del pacchetto:

```powershell
python scripts\validate_images.py --project-root .
```

Il risultato atteso è:

```text
{'valid': 6571}
```

Non ricostruire `final.parquet` o gli split se non sono cambiate le immagini: gli
artefatti già inclusi definiscono il campione preregistrato da 5.628 immagini.

## 2. Creazione degli archivi

Installa il progetto e crea i due archivi:

```powershell
python -m pip install -e .
python scripts\colab.py --project-root . bundle --output-dir colab_bundle
```

Otterrai:

```text
colab_bundle/colab_code.tar.gz
colab_bundle/colab_dataset.tar.gz
colab_bundle/colab_quickstart.ipynb
```

## 3. Caricamento su Google Drive

Crea in Drive:

```text
MyDrive/tesi_qml_colab/
```

Carica in quella directory `colab_code.tar.gz` e `colab_dataset.tar.gz`. Apri
`colab_quickstart.ipynb` in Colab, seleziona una GPU e avvia le celle nell'ordine
proposto. Il notebook copre l'intero ciclo operativo:
- primo run di prova e verifica persistenza su Drive;
- esecuzione riprendibile della matrice primaria full (160 run, regolabile con `FULL_RUN_LIMIT`);
- esecuzione riprendibile della campagna secondaria sui subset (100, 300, 900) tramite `scripts/colab_subsets.py`;
- aggregazione generale finale, calcolo statistiche e generazione di tutte le figure della tesi.

## 4. Ripresa delle sessioni e persistenza dello stato

Al termine di ogni run completato, i risultati e le feature computate vengono salvati atomicamente su Google Drive nei rispettivi archivi dedicati:

- **Matrice primaria full (160 run)**:
  ```text
  MyDrive/tesi_qml_colab/colab_state.tar.gz
  ```
- **Campagna secondaria subset (480 run)**:
  ```text
  MyDrive/tesi_qml_colab/colab_subset_state.tar.gz
  ```

Una nuova sessione Colab ripristina automaticamente lo stato corrispondente, saltando i run già completati e riprendendo l'esecuzione dal punto esatto di interruzione. Non sovrascrivere o eliminare manualmente tali archivi durante un'esecuzione.

## 5. Output per la tesi

Al completamento delle run pianificate, la cella finale di aggregazione unificata in `colab_quickstart.ipynb` richiama l'analisi statistica e genera tutte le figure vettoriali:
- metriche aggregate e test di permutazione Holm-Bonferroni per la matrice primaria full;
- curve di apprendimento e confronti secondari per la campagna subset;
- grafici in formato PDF salvati in `figures/`.

Tutti gli artefatti persistenti salvati in `results/` sono descritti formalmente in [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md).

Il pacchetto installa ed esegue PennyLane con PyTorch. Qiskit non è richiesto.
