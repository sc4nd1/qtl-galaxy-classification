# Guida operativa per Google Colab

Il pacchetto `tesivf_colab_tesi` contiene codice, manifest, split e
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
colab_bundle/colab_runner.ipynb
```

## 3. Caricamento su Google Drive

Crea in Drive:

```text
MyDrive/tesi_qml_colab/
```

Carica in quella directory `colab_code.tar.gz` e `colab_dataset.tar.gz`. Apri
`colab_quickstart.ipynb` in Colab, seleziona una GPU e avvia le celle nell'ordine
proposto. Il quickstart esegue un solo run e verifica il salvataggio.

Quando il primo run è riuscito, usa `colab_runner.ipynb`. La variabile
`SESSION_RUN_LIMIT` stabilisce quanti run eseguire nella sessione.

## 4. Ripresa delle sessioni

Al termine di ogni run, risultati e feature vengono salvati atomicamente in:

```text
MyDrive/tesi_qml_colab/colab_state.tar.gz
```

Una nuova sessione Colab ripristina automaticamente questo archivio. Non
sovrascriverlo manualmente durante un'esecuzione.

## 5. Output per la tesi

Quando la matrice è completa, il runner richiama aggregazione e generazione delle
figure. Gli artefatti persistenti si trovano in `results` dentro
`colab_state.tar.gz`; la struttura è descritta in `docs/OUTPUT_SCHEMA.md`.

Questo pacchetto installa ed esegue PennyLane. Qiskit non è richiesto.
