# Tre implementazioni PennyLane per i dataset Galaxy

## Analisi della cartella Drive

Nel Drive risultano due cartelle chiamate `dataset`.

La cartella personale `dataset` contiene:

- `GalaxyProject`: immagini gia divise in `train` e `val`, con classi `elliptical`, `spiral`, `merger`.
- `GalaxyProject2`: stesso schema `train`/`val` a tre classi, piu un checkpoint `galaxy_hybrid_BEST.pth`.
- `GalaxyProject3`: sorgenti Galaxy Zoo 2 piu artefatti riproducibili:
  - `gz2_hart16.csv`, catalogo grande originale.
  - `gz2_filename_mapping.csv`, mapping fra identificativi e file immagine.
  - `images_gz2/images`, cartella immagini.
  - `dataset_splits`, con `train_subset_50.csv`, `train_subset_100.csv`, `train_subset_300.csv`, `train_subset_1000.csv`, `train_full.csv`, `test_set.csv`.

Gli split CSV in `dataset_splits` hanno colonne:

```text
image_id,label,t08_odd_feature_a24_merger_weighted_fraction
```

Questa struttura e binaria: `label=1` indica merger, `label=0` non-merger. E utile per un esperimento dedicato merger-vs-rest, ma non coincide pienamente con la pipeline principale a tre classi.

La cartella condivisa `dataset` contiene `Galaxy_Dataset_Experiment`, organizzato in `Fold_1` ... `Fold_4`; ogni fold contiene le tre classi `elliptical`, `spiral`, `merger`. Questa e la forma piu vicina agli split stratificati usati dalla tesi.

## Implementazioni disponibili

### 1. `PL`: angle-QNN base

Classe: `PennylaneAngleQNN`

Usa la testa ResNet da 512 feature, comprime a 4 o 8 angoli, applica `RY` per l'encoding, layer trainabili `RZ/RY`, catena lineare di `CNOT`, e misura gli osservabili `Z`.

Uso consigliato: confronto principale con Qiskit, perche mantiene la stessa specifica circuitale.

```bash
python scripts/train.py --model PL --qubits 4 --layers 2 --subset-size 300 --device cuda
```

### 2. `PL_REUP`: data re-uploading

Classe: `PennylaneReuploadingQNN`

Genera un blocco di angoli per ogni layer, quindi reinietta i dati prima di ogni blocco variazionale. A parita di qubit puo rappresentare funzioni piu ricche, ma e anche piu facile da sovra-addestrare sui subset piccoli.

Uso consigliato: ablation di espressivita su subset 100/300/900, mantenendo lo stesso backbone e gli stessi split.

```bash
python scripts/train.py --model PL_REUP --qubits 4 --layers 2 --subset-size 300 --device cuda
```

### 3. `PL_AMP`: amplitude encoding

Classe: `PennylaneAmplitudeQNN`

Proietta le feature ResNet in un vettore normalizzato di dimensione `2 ** n_qubits` e lo carica come stato quantistico con `AmplitudeEmbedding`, seguito da `StronglyEntanglingLayers`.

Uso consigliato: esperimento esplorativo fuori dal confronto angle-only principale. A 8 qubit usa 256 ampiezze e puo sfruttare piu informazione del collo di bottiglia 4/8D, ma cambia il protocollo sperimentale.

```bash
python scripts/train.py --model PL_AMP --qubits 4 --layers 2 --subset-size 300 --device cuda
```

## Scelta pratica

Per la tesi principale conviene usare `PL`. Per un capitolo di confronto architetturale, `PL_REUP` e la variante piu pulita perche resta nel mondo angle encoding. `PL_AMP` e interessante come confronto esplorativo, ma va marcata esplicitamente come fuori-protocollo rispetto alla matrice principale descritta nel README.
