# Specifica di progetto: classificazione quantistica binaria e triclasse di galassie

**Data:** 2026-08-27  
**Stato:** proposta approvata in conversazione, da revisionare nel presente documento  
**Destinatario:** Codex incaricato dell'implementazione nel progetto `tesi_finale`

## 1. Obiettivo

Estendere la pipeline esistente per valutare in modo controllato una Dressed Quantum
Circuit (DQC) nello stile di Mari et al. su immagini di galassie. L'esperimento deve
confrontare classificatori classici e quantistici su due task:

1. classificazione binaria `elliptical` contro `spiral`;
2. classificazione triclasse `elliptical`, `merger`, `spiral`, dove `merger` è la
   classe morfologicamente più difficile.

Il risultato scientifico non deve essere presentato come *quantum advantage*. La
conclusione ammessa è più circoscritta: stabilire se, su questo dataset e con questo
protocollo, la componente quantistica addestrabile produce un contributo misurabile
rispetto a controlli classici e quantistici congelati.

## 2. Domande di ricerca

Le domande principali sono:

1. La QNN addestrabile supera il controllo classico con lo stesso bottleneck a 4 o 8
   dimensioni?
2. La QNN addestrabile supera lo stesso circuito lasciato congelato
   all'inizializzazione?
3. Il passaggio dal task binario al task triclasse riduce maggiormente le prestazioni
   del modello quantistico, di quello classico, oppure di entrambi in misura simile?
4. L'aumento da 4 a 8 qubit produce un miglioramento ripetibile?
5. Qual è il comportamento specifico della classe `merger` in termini di precision,
   recall e F1?

## 3. Perimetro

### Incluso nel confronto principale

- backbone ResNet18 con pesi ImageNet, completamente congelata;
- feature da 512 dimensioni pre-calcolate;
- task binario e triclasse;
- baseline B0, B1, B2-4 e B2-8;
- QNN congelate QF-4 e QF-8;
- QNN addestrabili Q-4 e Q-8;
- angle encoding, profondità quantistica fissa pari a 2;
- simulazione ideale PennyLane;
- verifica di equivalenza PennyLane/Qiskit;
- dieci repeat seed e un solo split stratificato 50/25/25 per seed;
- dataset completo, senza learning curve per subset nel confronto principale.

### Fuori dal confronto principale

- EfficientNet-B0, Zoobot e AstroCLIP;
- fine-tuning della backbone;
- B3 come parte dell'inferenza primaria;
- data re-uploading, amplitude encoding e QCNN;
- profondità quantistiche 1 o 4;
- hardware quantistico reale, shot noise e rumore hardware;
- ricerca estesa degli iperparametri;
- trasferimento dei pesi dal task binario a quello triclasse.

B3, EfficientNet-B0 e profondità 4 possono essere eseguiti solo come esperimenti
secondari dopo il completamento della matrice principale. I loro risultati devono
essere riportati separatamente.

## 4. Dataset e label

Usare il manifest finale rigenerato e validato, composto da 5.628 immagini:

| Classe | Campioni |
|---|---:|
| elliptical | 1.876 |
| merger | 1.876 |
| spiral | 1.876 |

Conservare le soglie già approvate:

- `p_elliptical > 0.8`;
- `p_spiral > 0.8`;
- `p_merger > 0.4`.

Le righe senza una sola hard label, i duplicati discordanti e gli hash presenti in
classi diverse devono continuare a essere esclusi con audit persistito.

## 5. Task e split annidati

Definire due configurazioni esplicite:

```text
binary:
  class_names = [elliptical, spiral]
  class_to_index = {elliptical: 0, spiral: 1}

three_class:
  class_names = [elliptical, merger, spiral]
  class_to_index = {elliptical: 0, merger: 1, spiral: 2}
```

La tabella di split triclasse è la sorgente unica. Il task binario deve essere
ottenuto filtrando `merger` e rimappando gli indici, senza rigenerare gli split.
Ogni `objid` elliptical o spiral deve quindi appartenere allo stesso split nei due
task.

Mantenere l'attuale assegnazione a quattro fold, ma usare soltanto `iteration=1`
nel confronto principale. Le altre tre rotazioni restano disponibili per una futura
estensione e non entrano nelle statistiche primarie.

Per ogni repeat seed e `iteration=1`:

| Task | Train | Validation | Test |
|---|---:|---:|---:|
| binary | 1.876 | 938 | 938 |
| three_class | 2.814 | 1.407 | 1.407 |

Il binario e il triclasse sono esperimenti indipendenti. Il modello triclasse non
deve caricare pesi, optimizer o checkpoint del modello binario. La sola componente
condivisa è la ResNet18 congelata e la relativa procedura di estrazione delle
feature.

## 6. Repeat seed e unità statistica

Usare questi dieci repeat seed, fissati prima di osservare i risultati:

```text
[42, 314, 2718, 1618, 2026, 7, 73, 137, 1729, 4096]
```

Il repeat seed è l'unità inferenziale e deve determinare, tramite sub-seed derivati
e registrati:

- split dei dati;
- vista di augmentation usata per la feature bank di training;
- inizializzazione dei layer classici;
- inizializzazione del circuito;
- ordine del DataLoader;
- eventuali generatori interni delle librerie.

I sub-seed devono essere derivati con una funzione stabile basata su
`repeat_seed + namespace`, non con `hash()` di Python, che non è stabile tra
processi. Persistire ogni sub-seed nel `run_config.json`.

## 7. Backbone e feature bank

Usare `torchvision.models.resnet18` con
`ResNet18_Weights.IMAGENET1K_V1`, sostituendo `fc` con `Identity` per ottenere
feature da 512 dimensioni.

La backbone deve essere:

- completamente congelata;
- sempre in modalità `eval`;
- identica per tutti i modelli;
- esclusa dagli optimizer;
- identificata nella provenance della feature bank.

Validation e test usano center crop 224 e normalizzazione ImageNet deterministica.
Il train usa rotazione, crop e flip già definiti nel progetto. Per rendere il costo
compatibile con Colab Pro, il confronto principale usa **una vista pre-calcolata per
objid e repeat seed** (`train_views: 1`), non quattro righe per oggetto. Cambiare il
numero di viste dopo l'apertura dei risultati di test invalida il confronto.

Le feature possono essere condivise tra task filtrandole dopo il caricamento. Non
duplicare fisicamente le feature binary e three-class su Drive.

## 8. Architetture

Sia `C=2` per il binario e `C=3` per il triclasse.

### B0: baseline lineare

```text
512 -> Linear(512, C)
```

Misura la separabilità lineare delle feature ResNet.

### B1: baseline MLP

```text
512 -> Linear(512, 512) -> ReLU -> Dropout(0.3) -> Linear(512, C)
```

È la baseline classica non lineare forte.

### B2-q: controllo classico del bottleneck

Per `q in {4, 8}`:

```text
512 -> Linear(512, q) -> tanh(x) * pi -> Linear(q, C)
```

B2 è un controllo del collo di bottiglia. Non deve essere descritto come emulazione
classica del circuito e non deve ricevere una non linearità sin/cos aggiuntiva.

### QF-q: circuito congelato all'inizializzazione

Per `q in {4, 8}`:

```text
512
 -> Linear(512, q), addestrabile
 -> tanh(x) * pi
 -> angle encoding RY
 -> 2 layer [RZ, RY, catena lineare CNOT], pesi congelati
 -> q misure locali <Z>
 -> Linear(q, C), addestrabile
```

I parametri quantistici devono avere `requires_grad=False`, non devono essere
inclusi nell'optimizer e devono risultare bit-identici prima e dopo il training.
La derivata rispetto agli input del circuito deve restare disponibile, altrimenti
la proiezione `512 -> q` non può apprendere.

### Q-q: Dressed Quantum Circuit addestrabile

Identica a QF-q, ma con i parametri quantistici inclusi nell'optimizer.

La specifica del circuito è:

- input: una `RY(x_i)` per qubit;
- per ciascuno dei due layer: `RZ(theta_z)`, poi `RY(theta_y)` su ogni qubit;
- entanglement: CNOT `(0,1), (1,2), ..., (q-2,q-1)`;
- output: aspettazione Pauli-Z locale per ogni qubit;
- parametri quantistici: `2 * depth * q = 4q`, cioè 16 per q=4 e 32 per q=8;
- scaling input: `tanh(x) * pi`.

## 9. Inizializzazione appaiata

Per ogni tupla `(task, repeat_seed, iteration=1, q)`:

1. B2-q, QF-q e Q-q devono partire dagli stessi pesi e bias della proiezione
   `512 -> q`;
2. B2-q, QF-q e Q-q devono partire dagli stessi pesi e bias dell'head `q -> C`;
3. QF-q e Q-q devono partire dagli stessi identici pesi quantistici;
4. i pesi quantistici devono essere campionati da `Normal(0, 0.05^2)`;
5. QF congela tali pesi, Q li ottimizza;
6. inizializzazioni e checksum devono essere salvati nella configurazione del run.

Non affidarsi all'ordine implicito di costruzione dei moduli PyTorch: il circuito
consuma numeri casuali e può rendere diversi gli head finali. Implementare una
routine esplicita di inizializzazione appaiata o un piccolo oggetto contenente gli
state dict iniziali condivisi.

B0 e B1 devono usare lo stesso repeat seed e lo stesso ordine dei dati, ma non sono
architetturalmente appaiabili layer per layer.

## 10. Ottimizzazione

Per il confronto principale:

- loss: `CrossEntropyLoss` per entrambi i task;
- output: 2 o 3 logits, mai sigmoid singola;
- optimizer: AdamW;
- learning rate comune: `1e-3` per B0, B1, B2, QF e Q;
- weight decay: `1e-4`;
- massimo 80 epoche;
- early stopping sulla validation loss, patience 12, `min_delta=1e-4`;
- scheduler `ReduceLROnPlateau`, factor 0.5, patience 4;
- gradient clipping globale a 1.0;
- checkpoint migliore selezionato esclusivamente tramite validation loss;
- test sigillato fino al ripristino del checkpoint migliore;
- nessuna modifica degli iperparametri dopo aver aperto metriche o predizioni test.

Per B2, QF e Q usare lo stesso batch size, così la rumorosità dei gradienti è
comparabile. Prima della matrice, eseguire un benchmark tecnico di una sola epoca
su Q-8 triclasse, senza produrre metriche test, con batch size 8, 16 e 32. Scegliere
il più grande che completa forward/backward senza OOM e applicarlo a tutto il
trio B2/QF/Q. Questa scelta è operativa e deve avvenire prima di qualsiasi analisi
test. Registrare il batch size scelto nella configurazione bloccata.

Il benchmark tecnico non entra nei risultati della tesi.

## 11. Matrice principale

```text
2 task
x 10 repeat seed
x 1 split attivo per seed
x 8 configurazioni
= 160 run
```

Le otto configurazioni sono:

```text
B0
B1
B2-4
B2-8
QF-4
QF-8
Q-4
Q-8
```

La matrice deve essere materializzata prima dell'esecuzione, con identità univoca e
stato `pending`, `running`, `complete` o `failed`. Un run è completo solo quando
esistono configurazione, checkpoint best, history, metriche test e predizioni test.

## 12. PennyLane e Qiskit

PennyLane è l'implementazione usata per tutti i training QF e Q. Qiskit non è una
seconda famiglia di modelli e non deve duplicare i 160 run.

Qiskit serve a verificare, per q=4 e q=8:

- stessa topologia;
- stesso ordine delle porte;
- stesse osservabili;
- stesse aspettazioni su input e pesi fissati;
- gradienti rispetto agli input compatibili;
- gradienti rispetto ai pesi compatibili per Q;
- convenzione coerente sull'ordine dei qubit.

Questi controlli appartengono ai test automatici e a un piccolo artefatto di
validazione, non alla matrice statistica.

## 13. Metriche

### Endpoint primario

- macro-F1 sul test.

### Endpoint secondari

- accuracy;
- balanced accuracy;
- precision, recall e F1 macro/weighted/micro;
- precision, recall e F1 per classe;
- confusion matrix grezza e normalizzata;
- ROC-AUC one-vs-rest;
- PR-AUC;
- log loss;
- Brier score;
- expected calibration error;
- runtime, epoca migliore, memoria, parametri totali e addestrabili;
- norma e varianza dei gradienti quantistici per Q;
- F1 e recall di `merger` evidenziati nel task triclasse.

## 14. Analisi statistica preregistrata

Ogni confronto deve essere appaiato per task, repeat seed, split e q. L'unità
inferenziale è il repeat seed (`n=10`). Non trattare i campioni o le epoche come
repliche indipendenti.

### Famiglia primaria di ipotesi

Applicare correzione Holm a questi quattro confronti bilaterali sulla macro-F1:

1. binary: Q-4 meno B2-4;
2. binary: Q-8 meno B2-8;
3. three_class: Q-4 meno B2-4;
4. three_class: Q-8 meno B2-8.

Per ogni confronto riportare:

- differenza media appaiata;
- deviazione standard delle differenze;
- intervallo bootstrap al 95% sulla media;
- permutation test a inversione di segno;
- p-value grezzo e corretto Holm;
- Cohen's dz;
- Wilcoxon bilaterale come analisi di sensibilità.

### Confronti secondari

- Q-q contro QF-q;
- QF-q contro B2-q;
- Q-8 contro Q-4;
- Q contro B1;
- differenza binario meno triclasse per ciascun modello;
- metriche specifiche dei merger.

I confronti secondari sono esplorativi e devono essere etichettati come tali.

## 15. Schema di interpretazione

Pre-registrare almeno i seguenti casi:

- `Q ~= B2`: nessuna evidenza che il circuito migliori il controllo classico;
- `QF ~= B2 < Q`: il contributo emerge dall'ottimizzazione quantistica;
- `B2 < QF ~= Q`: la feature map quantistica è utile, ma addestrarla non aggiunge;
- `QF < B2` e `Q ~= QF`: il circuito congelato distrugge informazione e il training
  non la recupera;
- `QF < Q < B2`: il training recupera parte dell'informazione, senza raggiungere
  il controllo classico;
- `QF < B2 < Q`: il circuito congelato danneggia le feature, ma l'ottimizzazione
  recupera e supera B2;
- `Q <= B1`: nessun miglioramento rispetto alla baseline classica non lineare;
- `Q > B1`: il modello ibrido performa meglio su questo task, senza affermare
  vantaggio computazionale o quantum advantage.

`~=` indica compatibilità entro incertezza ed effect size osservati, non uguaglianza
matematica.

## 16. Esecuzione su Google Colab Pro

Il notebook Colab deve essere un orchestratore sottile. La logica scientifica deve
restare nel package e negli script testabili.

Flusso di una sessione:

1. montare Google Drive;
2. estrarre codice e dataset in `/content`, evitando training direttamente sugli
   archivi compressi;
3. installare versioni pin delle dipendenze;
4. validare fingerprint di manifest, split e configurazioni;
5. sincronizzare o generare le feature mancanti;
6. leggere la matrice e individuare il prossimo run incompleto;
7. riprendere da `last.pt` quando il fingerprint coincide;
8. salvare su Drive checkpoint e artefatti atomici dopo ogni epoca/run;
9. saltare automaticamente i run completi;
10. produrre a fine sessione un riepilogo di completati, falliti e mancanti.

Non assumere una durata garantita della VM. Ogni run deve essere riprendibile dopo
un'interruzione. Non affidarsi a processi in background, SSH o meccanismi che
tentino di aggirare i limiti Colab.

L'uso della GPU deve essere misurato. Se il simulatore PennyLane selezionato non
utilizza effettivamente CUDA, il runner deve segnalarlo e permettere il runtime CPU
senza cambiare i risultati scientifici. La scelta CPU/GPU è operativa e va registrata.

## 17. Persistenza e riproducibilità

Ogni run deve salvare almeno:

- `run_config.json` completo;
- fingerprint della configurazione;
- task e mapping delle classi;
- repeat seed e tutti i sub-seed;
- checksum del manifest e della feature bank;
- modello, q, depth e flag quantum frozen;
- checksum dei pesi iniziali condivisi;
- optimizer, scheduler e batch size;
- `checkpoints/best.pt` e `checkpoints/last.pt`;
- `history.jsonl`;
- metriche validation e test;
- predizioni test per objid;
- conteggio parametri totali e addestrabili;
- firma del circuito;
- versioni Python, PyTorch, torchvision, PennyLane e Qiskit;
- tipo di runtime e device effettivo.

Il resume deve rifiutare checkpoint con fingerprint incompatibile. In QF salvare e
verificare il checksum dei pesi quantistici prima e dopo il training.

## 18. Modifiche richieste al codice

Questa sezione descrive il comportamento richiesto; Codex deve mantenere moduli
piccoli e testabili e seguire le convenzioni già presenti nel repository.

### Configurazioni

- `configs/dataset.yaml`
  - portare `repeat_seeds` ai dieci valori fissati;
  - dichiarare `primary_iterations: [1]`;
  - aggiungere configurazioni esplicite per `binary` e `three_class`.
- `configs/training.yaml`
  - impostare `feature_bank.train_views: 1` per il confronto principale;
  - dichiarare `quantum_init_std: 0.05`;
  - dichiarare matrice principale e iperparametri bloccati;
  - rimuovere i subset dalla matrice primaria senza eliminare il supporto legacy.

### Task e feature

- introdurre un modulo dedicato, per esempio `src/tesi_qml/tasks.py`, che restituisca
  class names, mapping e filtro per ogni task;
- aggiungere una funzione pura che filtri una `FeatureBank` e rimappi le label senza
  alterare objid, view o provenance;
- includere task e mapping nella provenance/fingerprint;
- impedire label non contigue dopo il filtro binario.

### Modelli

- rendere `num_classes` un parametro passato esplicitamente dal task a ogni modello;
- aggiungere QF usando la stessa implementazione di Q, con un flag o costruttore
  esplicito che congeli soltanto i pesi quantistici;
- assicurare che il circuito resti differenziabile rispetto agli input in QF;
- introdurre una routine di inizializzazione appaiata per B2/QF/Q;
- conservare la firma circuitale comune PennyLane/Qiskit;
- riportare separatamente parametri totali, addestrabili e quantistici.

### Training

- `scripts/train.py`
  - aggiungere `--task {binary,three_class}`;
  - aggiungere il modello `QF`;
  - usare class names e probability columns dinamiche;
  - filtrare/remappare feature bank prima di creare i DataLoader;
  - includere task, freeze state e initialization checksum nel run name e fingerprint;
  - creare optimizer solo con parametri `requires_grad=True`;
  - usare uno split attivo per repeat seed nella matrice primaria.
- mantenere il test sigillato fino al ripristino del checkpoint migliore;
- verificare a fine QF che i pesi quantistici siano immutati.

### Runner della matrice e Colab

- aggiungere uno script testabile, per esempio `scripts/run_matrix.py`, che:
  - materializzi esattamente 160 identità di run;
  - mostri un dry-run senza addestrare;
  - esegua il prossimo run pending;
  - supporti un limite di run per sessione;
  - riprenda run interrotti;
  - non rilanci run completi;
  - produca un report di copertura della matrice.
- aggiornare `notebooks/colab_runner.ipynb` perché chiami esclusivamente script e
  package, senza duplicare logica di modelli o metriche nel notebook;
- aggiornare bundle e restore dei risultati su Drive.

### Aggregazione e statistica

- rendere task e q colonne di raggruppamento obbligatorie;
- validare una matrice 10 seed x iteration 1 per ciascuna configurazione;
- aggiungere correzione Holm ai quattro confronti primari;
- esportare p-value grezzi/corretti, effect size e CI;
- produrre tabelle specifiche per binary, three-class e merger;
- impedire l'aggregazione di run incompleti o con fingerprint incompatibili.

### Figure

Produrre almeno:

1. schema ResNet18 -> bottleneck -> B2/QF/Q;
2. macro-F1 per modello e task con CI 95%;
3. differenze appaiate Q-B2 e Q-QF per seed;
4. confronto 4 contro 8 qubit;
5. confusion matrix triclasse;
6. precision/recall/F1 dei merger;
7. training curve rappresentative;
8. runtime e numero di parametri.

## 19. Test richiesti

### Unit test

- task binary filtra i merger e rimappa `elliptical=0`, `spiral=1`;
- task three-class conserva `0,1,2`;
- gli objid elliptical/spiral mantengono lo stesso split nei due task;
- ogni primary seed usa una sola iteration;
- B0/B1/B2/QF/Q producono 2 o 3 logits corretti;
- B2 e Q condividono dimensione del bottleneck;
- B2/QF/Q ricevono identici pesi iniziali nei layer condivisi;
- QF/Q ricevono identici pesi quantistici iniziali;
- QF non modifica i pesi quantistici durante optimizer step o training;
- QF produce gradienti non nulli per proiezione e head;
- Q produce gradienti finiti per proiezione, circuito e head;
- PennyLane e Qiskit concordano su aspettazioni e gradienti;
- probability columns seguono dinamicamente le class names;
- fingerprint cambia con task, q, freeze flag, seed o mapping;
- Holm correction restituisce valori corretti e monotoni;
- validazione matrice rifiuta duplicati, mancanti ed extra.

### Integration test

- mini-pipeline completa binary con B2 e QF;
- mini-pipeline completa three-class con Q;
- resume produce lo stesso risultato di un training non interrotto;
- dry-run della matrice produce esattamente 160 run univoci;
- un run completo viene saltato;
- un run parziale viene ripreso;
- aggregazione usa soltanto run completi;
- bundle Colab include codice/config/dataset richiesti ed esclude cache inutili.

## 20. Gestione degli errori

Il codice deve fallire presto e con messaggi chiari quando:

- task o classe non sono supportati;
- le label filtrate non sono contigue;
- feature e metadata non sono allineati;
- uno split contiene leakage o copertura incompleta;
- manca una classe in train, validation o test;
- QF contiene parametri quantistici nell'optimizer;
- i pesi congelati cambiano;
- compaiono NaN/Inf in logits, loss o gradienti;
- un checkpoint appartiene a un'altra configurazione;
- Qiskit e PennyLane divergono oltre la tolleranza;
- Drive non è scrivibile o lo spazio è insufficiente;
- la matrice contiene run duplicati o incompleti.

Gli artefatti devono essere scritti prima in file `.part` e poi rinominati. Non
cancellare automaticamente checkpoint o risultati incompleti: devono restare
recuperabili o diagnosticabili.

## 21. Criteri di accettazione

L'implementazione è completa quando:

1. tutti i test non quantistici passano;
2. tutti i test quantistici PennyLane/Qiskit passano;
3. le mini-pipeline binary e three-class passano;
4. il dry-run mostra esattamente 160 run primari univoci;
5. B2/QF/Q sono appaiati nell'inizializzazione come specificato;
6. QF conserva esattamente i pesi quantistici;
7. ogni run è riprendibile dopo interruzione;
8. metriche e predizioni includono task, class mapping, seed e objid;
9. l'aggregazione produce confronti appaiati, CI, effect size e Holm correction;
10. il notebook Colab esegue la pipeline senza contenere logica scientifica duplicata;
11. README e documentazione descrivono correttamente limiti e framing scientifico;
12. nessuna conclusione usa impropriamente l'espressione `quantum advantage`.

## 22. Ordine logico dell'implementazione

Il successivo piano operativo per Codex deve scomporre il lavoro in passi piccoli,
ciascuno con test scritto prima del codice di produzione. L'ordine raccomandato è:

1. task configuration e filtro/remap delle feature;
2. output dinamico a 2/3 classi;
3. inizializzazione appaiata;
4. modello QF e invarianti sui pesi congelati;
5. identità dei run e matrice da 160 elementi;
6. runner riprendibile per Colab;
7. aggregazione statistica e Holm;
8. figure e documentazione;
9. verifica end-to-end e bundle Colab.

Questa sezione stabilisce dipendenze e priorità, ma non sostituisce il piano di
implementazione dettagliato con percorsi, test e comandi di verifica.
