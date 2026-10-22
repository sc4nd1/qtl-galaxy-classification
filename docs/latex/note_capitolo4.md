# Capitolo 4 — audit editoriale e tecnico

Data: 7 settembre 2026.

## Valutazione complessiva

**Bozza condivisibile con le cautele esplicitate nel testo.**
Il metodo è stato confrontato con codice, archivio del codice, notebook eseguiti
e metadati delle 640 run in colab_pacchetto_finale.
Questo audit non è una nuova esecuzione del training e non è una certificazione
di validità inferenziale rispetto a campioni astronomici indipendenti.

## Fonti interne e controlli

| Affermazione | Riscontro |
|---|---|
| 640 run, 160 per regime | run_config.json e quattro matrici, tutte complete |
| Metriche di test presenti | 640 test_metrics.json |
| 10 seed, iteration=1 | metadati di tutte le run |
| Struttura B0/B1/B2 | src/tesi_qml/models/classical.py e scripts/train.py |
| Encoding RY, tanh_pi, L=2, catena CNOT aperta | pennylane_qnn.py, circuit_spec.py e model_signature in metrics.json |
| Conteggi di parametri | runtime nei 640 test_metrics.json; conteggi coerenti in ogni configurazione |
| Inizializzazione appaiata | initialization.py e initialization_checksums |
| 160 gruppi B2/Q/QF appaiati | checksum, sotto-seed condivisi e provenance train/validation/test uguali entro gruppo |
| Batch 8 per B2/Q/QF; 32 per B0/B1 | ramificazione effettiva in scripts/train.py |
| Ottimizzazione e arresto | training_config, engine.py, checkpointing.py |
| Metriche di prestazione finali dal test | scripts/aggregate_results.py: legge test_metrics.json |
| IC delle figure finali bootstrap | sorgenti di 04_colab_results_analysis.ipynb; 06 legge le tabelle prodotte da 04 |
| Test primari e secondari | primary_hypotheses.json e secondary_subset_hypotheses.json |

Gli hash dei seguenti file nel colab_code.tar.gz del pacchetto coincidono con
quelli del progetto locale: classical.py, circuit_spec.py, pennylane_qnn.py,
initialization.py, engine.py, training/metrics.py, evaluation/statistics.py e
scripts/train.py. Questo non esclude le patch runtime del notebook: per i subset
la provenance delle run e il notebook restano indispensabili.

## Verifica indipendente dei test salvati

Ricalcolati dalle Macro-F1 dei singoli test_metrics.json, mantenendo
ordinamento dei seed e random_seed dichiarato in ciascun confronto:

- 4 test primari: differenza media, bootstrap percentile, sign-flip Monte Carlo
  e Holm coincidono con i valori archiviati;
- 12 test secondari: stessi controlli, tutti coincidenti;
- 10 coppie per test, 10.000 bootstrap, 10.000 sign-flip;
- chiavi delle 640 run senza duplicati.

La coincidenza numerica non verifica le assunzioni statistiche del test.
Nel capitolo sono esplicitati dipendenza tra holdout, simmetria per il sign-flip
e distinzione tra intervalli marginali e correzione della molteplicità.

Comandi dell'audit locale (richiedono il repository completo):

    py -3.10 tmp/chapter4_audit.py runs
    py -3.10 tmp/chapter4_audit.py stats
    py -3.10 tmp/chapter4_audit.py papers

Lo script è di sola lettura sugli input: non modifica risultati o archivi.

## Discrepanze rispetto agli appunti storici

FormalizzazioneModelli.pdf (pagine 1–4) descrive una versione diversa:

| Appunto storico | Matrice finale |
|---|---|
| Input 256×256 | Crop 224×224 dal cutout 320×320 |
| Scaling pi/2 | pi * tanh |
| Encoding RX | Encoding RY |
| BasicEntanglerLayers, L=1 | RZ poi RY, L=2 |
| CNOT ad anello | Catena aperta 0→1→…→q−1 |
| Quattro fold e subset 50 | Repeated holdout, iteration=1; subset 100/300/900 |
| Risultati del vecchio dataset | 640 run finali distinte |

Non sono state trasferite nel nuovo capitolo le affermazioni storiche
di “prova della local effective dimension”, “perdita topologica” o
“vantaggio garantito sui pochi dati”: non sono misure effettuate nel protocollo
finale. Il documento storico rimane intatto e può servire per ricostruire
l'evoluzione del progetto, non per descrivere la versione finale.

## Cautele importanti per il capitolo 5

1. **Hardware.** Tutte le 160 run full riportano CUDA/Tesla T4.
   Tutte le 480 run dei subset riportano CPU. Nessun gruppo appaiato
   B2/Q/QF mescola CPU e CUDA. Non interpretare le traiettorie temporali
   100→300→900→Full come scaling a hardware costante.
2. **Inferenza.** inference_seconds nelle tabelle correnti proviene dalla
   validation; test_inference_seconds è distinto. Le prestazioni invece sono
   quelle del test. Non chiamare il primo campo “latenza del test”.
3. **Training.** Somma dei tempi di epoca, non wall time della sessione e
   non costo end-to-end della pipeline. La feature extraction è esclusa.
4. **Gradienti.** quantum_gradient_norm usa il nome del parametro e include
   pre_quantum/post_quantum: non isola i pesi variazionali.
5. **QF.** Congela solo i pesi circuitali; il gradiente verso l'input resta necessario.
6. **Intervalli.** I summary intermedi possono contenere IC t; le figure 04/06
   usano bootstrap. Non mescolare definizioni senza specificarlo.
7. **PR.** pr_auc è average precision, non area trapezoidale.
8. **Capacità.** Numero di pesi simile non implica identica capacità;
   4→8 cambia anche proiezione e classificatore.
9. **Preregistrazione.** Il capitolo non dichiara una preregistrazione pubblica.
10. **Confronti esplorativi.** Q−QF e differenza-in-differenze non identificano
    automaticamente effetti causali dell'entanglement o della classe merger.

Le cautele 1–2 richiedono attenzione nelle future didascalie dei grafici;
in questo turno non sono stati modificati notebook o risultati.

## Prossima revisione con l'autore

Leggere prima:
- sezione del circuito, per verificare che il livello di spiegazione sia adeguato;
- paragrafo capacità comparabile e ruolo di QF;
- sezione dei costi, per concordare come presentare CPU e GPU nei risultati.

Il capitolo 3 non è stato modificato. La bozza 4 introduce le precisazioni
sull'ambiente effettivo senza riscrivere retroattivamente i dati sperimentali.

## Verifica del documento

Controlli statici di include, citazioni, label, ambienti e parentesi superati
su tutti e quattro i documenti principali/anteprime, senza errori rilevati.
Bibliografia: 14 voci complessive; capitolo 4: due figure e quattro tabelle.
Non è disponibile un motore TeX locale: impaginazione e due schemi TikZ
devono essere compilati e controllati su Overleaf.
