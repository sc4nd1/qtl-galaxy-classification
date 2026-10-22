# Protocollo di partizionamento e validazione sperimentale

Il dataset finale contiene 5.628 immagini bilanciate: 1.876 per ciascuna delle
classi `elliptical`, `merger` e `spiral`. Le assegnazioni non sono rappresentate
da copie fisiche in cartelle `Fold_1`, `Fold_2`, ecc.; la sorgente unica è
`tesi_finale/data/splits/all.parquet`.

## 1. Quattro fold stratificati

Per ogni repeat seed, ciascuna classe viene suddivisa in quattro fold da 469
campioni. Ogni fold contiene quindi:

| Classe | Campioni per fold |
|---|---:|
| `elliptical` | 469 |
| `merger` | 469 |
| `spiral` | 469 |
| **Totale** | **1.407** |

Le quattro rotazioni sono:

| Iterazione | Test | Validation | Train | Train totale |
|---:|---|---|---|---:|
| 1 | Fold 4 | Fold 3 | Fold 1 + Fold 2 | 2.814 |
| 2 | Fold 1 | Fold 4 | Fold 2 + Fold 3 | 2.814 |
| 3 | Fold 2 | Fold 1 | Fold 3 + Fold 4 | 2.814 |
| 4 | Fold 3 | Fold 2 | Fold 4 + Fold 1 | 2.814 |

In ogni iterazione del task triclasse si hanno 2.814 immagini di train, 1.407 di
validation e 1.407 di test. Ogni ObjID riceve una sola assegnazione per coppia
`repeat_seed`/`iteration`.

## 2. Dieci repeat seed

Gli split sono generati con i seed preregistrati:

```text
[42, 314, 2718, 1618, 2026, 7, 73, 137, 1729, 4096]
```

Il file `all.parquet` contiene 225.120 righe:

```text
5.628 immagini × 10 seed × 4 iterazioni = 225.120 assegnazioni
```

Sono quindi disponibili 40 rotazioni stratificate complete. Tutti i modelli
confrontati devono usare le stesse assegnazioni per assicurare confronti appaiati.

## 3. Protocollo primario della tesi

I quattro fold descrivono la struttura disponibile, ma il confronto principale
non tratta le quattro rotazioni dello stesso seed come repliche indipendenti. Per
ogni repeat seed viene usata soltanto `iteration=1`:

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| Binario (`elliptical`, `spiral`) | 1.876 | 938 | 938 |
| Triclasse | 2.814 | 1.407 | 1.407 |

Il task binario è ottenuto filtrando la classe `merger` dallo split triclasse e
rimappando le due classi. Non vengono rigenerati split separati: uno stesso ObjID
`elliptical` o `spiral` mantiene la medesima assegnazione nei due task.

La matrice primaria contiene:

```text
2 task × 10 repeat seed × 1 iterazione × 8 configurazioni = 160 run
```

Per ora i modelli quantistici e quantum-frozen sono implementati ed eseguiti con
PennyLane. Il dataset e gli split non dipendono dal backend quantistico.

## 4. Dataset completo e studi secondari

Il confronto principale usa l'intero training set disponibile. I subset da 50,
100, 300 e 900 immagini descritti nella metodologia storica non fanno parte della
matrice primaria e non devono essere mescolati con i risultati principali. Possono
essere reintrodotti in futuro come analisi esplorativa separata, mantenendo il
campionamento stratificato e dichiarando esplicitamente la natura secondaria.

## 5. Inferenza statistica

L'unità inferenziale è il repeat seed (`n = 10`), non la singola immagine, epoca o
rotazione. Ogni confronto è appaiato per task, seed, split e numero di qubit.

La famiglia primaria comprende quattro confronti bilaterali sulla macro-F1 tra il
modello PennyLane Q e la baseline B2, per task binario/triclasse e 4/8 qubit. Per
ogni confronto vengono riportati:

- differenza media appaiata e deviazione standard;
- intervallo bootstrap al 95% sulla media;
- permutation test a inversione di segno;
- p-value grezzo e corretto con procedura di Holm;
- Cohen's dz;
- test di Wilcoxon bilaterale come analisi di sensibilità.

Il Welch t-test della metodologia storica non è il test primario, perché i run
sono appaiati sullo stesso seed e sullo stesso split.

## 6. Regole contro il data leakage

- Il test set non partecipa a training, selezione dell'epoca o tuning.
- La validation seleziona checkpoint e iperparametri operativi.
- Baseline e modelli PennyLane usano gli stessi ObjID per ogni split.
- Il manifest triclasse è la sorgente unica anche per il task binario.
- I risultati delle quattro iterazioni disponibili non vengono presentati come 40
  repliche indipendenti nel protocollo primario.
