# Capitolo 5 — audit dei risultati e note per la revisione

Data: 7 settembre 2026.

## Fonte dei numeri

Fonte primaria: `colab_pacchetto_finale`, non una concatenazione indiscriminata
delle copie presenti in `results`.

- `colab_state/results/runs`: 160 run Full.
- `colab_subset_state/results/runs`: 480 run dei subset 100/300/900.
- `test_metrics.json`: prestazioni di test, metriche per classe,
  matrici di confusione normalizzate e campi runtime.
- `run_config.json`: task, modello, dimensione, seed, regime, iterazione e dispositivo.
- `colab_state/results/aggregate/primary_hypotheses.json`: quattro test primari.
- `colab_subset_state/results/aggregate/secondary_subset_hypotheses.json`:
  dodici test secondari.
- `figure_restyle/figures`: sei PDF finali, verificati visivamente tramite
  le corrispondenti PNG; esclusi i due grafici delle vecchie versioni.

Attenzione: `colab_subset_state/results/aggregate/runs.parquet` ha **640 righe**,
incluse le 160 Full, mentre il riepilogo Full ne ha 160. Concatenarli senza
deduplicare produrrebbe 800 righe e un doppio conteggio del Full.

## Controlli numerici e riproducibilità

Gli script locali di audit richiedono Python, NumPy e pandas con supporto Parquet:

    py -3.10 tmp/chapter4_audit.py stats
    py -3.10 tmp/chapter5_audit.py

Non addestrano modelli e non modificano dati, archivi o notebook.

Il secondo ricostruisce 640 chiavi uniche (task, subset, modello, qubit,
repeat_seed, iterazione), dieci seed in ogni configurazione e 160 run per regime.
Confronta ciascun riepilogo Parquet con i JSON senza duplicare il Full:
Macro-F1, ECE, Brier score e i due campi di tempo preesistenti coincidono.
Gli intervalli delle prestazioni assolute sono bootstrap percentile della
media su 10 seed, con 10.000 ricampionamenti e generatore deterministico
definito nello script; non sono gli IC t dei vecchi summary.

I quattro test primari e i dodici secondari sono riportati con gli IC e i
p-value archiviati; lo script del capitolo 4 li ricalcola dai JSON, inclusi
sign-flip e Holm. Non è stata introdotta una terza famiglia di test.

Differenza-in-differenze e Q−QF sono ricostruiti per seed prima del bootstrap,
con la convenzione di seed deterministico del notebook 06. Gli IC nelle
tabelle/testo possono differire leggermente dalle figure per arrotondamento
CSV, ordine dei record o seed di ricampionamento. L'audit lavora sulla
precisione originale dei JSON. Le cifre riportate nel testo sono arrotondate.

## Scelte editoriali e precisazioni

1. Prestazioni assolute: entrambe le tabelle includono tutti gli otto modelli
   e tutti i quattro regimi, con media e IC95%; nessuna selezione del seed migliore.
2. Holm: quattro confronti Full e dodici subset costituiscono famiglie separate.
   Un IC marginale che esclude zero può accompagnare un p-value Holm > 0,05.
3. Il divario Q−B2 a otto qubit nel triclasse **si riduce** da circa −3,30 pp
   a −1,67 pp aumentando i dati. Il titolo del grafico scaling suggeriva
   impropriamente il contrario; non è ripreso nel testo.
4. La differenza-in-differenze è una tabella: nel PDF `03_binary_to_three_class`
   l'asse inferiore è fissato a −5 pp, mentre il limite bootstrap del subset
   100/8q è circa −5,17 pp. La tabella conserva l'intero intervallo; nessun
   dato viene eliminato e il PDF originale non è modificato.
5. Merger: sul Full/8q contribuisce a circa metà del divario Macro-F1; sul
   Full/4q la maggiore perdita di F1 per classe riguarda invece le spirali.
   Non sostenere che tutte le penalità siano soltanto un problema di merger.
6. Matrici di confusione: prima normalizzazione per riga di ciascun seed,
   poi media delle dieci matrici. Le percentuali citate sono tassi medi
   condizionati alla classe vera, non conteggi di galassie indipendenti.
   Non è stata aggiunta una nuova figura di confusion matrix per-run.
7. Calibrazione: tabella descrittiva Full di B2 e Q; nessuna reliability
   curve singola scelta come prova generale. ECE minore non implica
   Macro-F1 o log-loss migliore (esempio Q-8 triclasse).
8. Costi: Full GPU/Tesla T4, subset CPU. La traiettoria tra regimi non è
   una frontiera a hardware costante. Nel binario non si deve affermare
   una dominanza Pareto stretta di B2 su ogni punto quantistico.
9. La tabella inferenza usa **test_inference_seconds**, non il campo della
   validation `inference_seconds`. Il training è somma dei tempi di epoca,
   non wall time Colab, né costo end-to-end comprensivo della feature cache.
10. I rapporti Q/B2 sono mediane dei rapporti appaiati, non rapporti di mediane.
11. Nessuna nuova conclusione sui barren plateau: la metrica storica del
    gradiente non isola i soli parametri circuitali.

## Figure incluse

Copie integrali dei PDF originali in `figures/results`:

- `01_hero_learning_curves.pdf`
- `02_delta_q_minus_b2.pdf`
- `04_merger_detailed.pdf`
- `05_q_minus_qf.pdf`
- `06_qubit_sensitivity.pdf`
- `07_pareto_performance_cost.pdf`

Nel sorgente LaTeX, `trim=0 0 0 30bp,clip` omette soltanto il margine
superiore con il titolo editoriale incorporato. Assi, dati e legende
rimangono nella figura; la didascalia della tesi fornisce la lettura
neutra e le limitazioni pertinenti. I PDF archiviati sono intatti.

## Stato e revisione successiva

Bozza del capitolo completa, con sei figure e nove tabelle. Il capitolo 6
rimane da scrivere: non sono state trasferite qui spiegazioni generali
su expressivity, rumore delle etichette o vantaggio quantistico.

Non è disponibile un motore LaTeX locale: la compilazione e l'impaginazione
non sono certificate dai soli controlli statici. Su Overleaf controllare
in particolare leggibilità delle figure a due pannelli, eventuali tabelle
troppo larghe e posizione dei float.

Su successiva richiesta dell'autore, la bibliografia dei capitoli 4 e 5
è stata integrata dopo la stesura dei risultati. Sono stati aggiunti
richiami a Nadeau e Bengio, Efron, Holm, Guo et al. e Gneiting e Raftery
nei passaggi metodologici del capitolo 5. Le cifre dell'esperimento
restano attribuite agli artefatti, non a questi articoli. Dettagli e
riscontri in `bibliografia_capitoli4_5.md`.
