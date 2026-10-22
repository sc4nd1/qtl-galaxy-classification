# Esecuzione della campagna sperimentale su Google Colab

## Obiettivo e organizzazione generale

La campagna sperimentale è stata eseguita su Google Colab per disporre di un
ambiente Linux con accelerazione GPU, senza dipendere dalla persistenza della
macchina virtuale assegnata da Colab. La caratteristica principale della
procedura è infatti la **riprendibilità**: ogni esperimento completato, insieme
alle feature già calcolate, viene salvato su Google Drive e può essere
ripristinato in una sessione successiva.

La matrice sperimentale primaria comprende 160 addestramenti indipendenti:

```text
2 task × 10 repeat seed × 1 iterazione × 8 configurazioni = 160 run
```

I due task sono la classificazione binaria delle galassie ellittiche e spirali e
la classificazione triclasse di galassie ellittiche, merger e spirali. Per ogni
task vengono confrontate tre baseline classiche, indicate con B0, B1 e B2, e le
configurazioni ibride quantum-classical QF e Q con 4 oppure 8 qubit. B2 è
presente nelle varianti a 4 e 8 dimensioni di bottleneck, così da permettere un
confronto controllato con i modelli quantistici corrispondenti.

La matrice usa i dieci repeat seed preregistrati e soltanto `iteration=1` delle
quattro rotazioni disponibili. Di conseguenza, il repeat seed costituisce
l'unità inferenziale e i confronti fra modelli sono appaiati sullo stesso split.

## Preparazione del dataset e dei bundle

Prima del trasferimento su Colab è stata verificata localmente la struttura del
dataset fisico. L'archivio contiene 6.571 immagini JPEG da 320×320 pixel,
suddivise in:

| Classe | Immagini valide |
|---|---:|
| Ellittiche | 2.347 |
| Merger | 1.877 |
| Spirali | 2.347 |
| **Totale** | **6.571** |

Da queste immagini, le procedure di pulizia, deduplicazione e bilanciamento
hanno definito il campione sperimentale finale di 5.628 immagini, con 1.876
elementi per classe. Il dataset finale e gli split non vengono ricostruiti su
Colab: vengono trasferiti gli artefatti già validati, in modo che ogni sessione
usi esattamente lo stesso campione preregistrato.

Per il trasferimento sono stati prodotti due archivi distinti:

- `colab_code.tar.gz`, contenente codice sorgente, configurazioni, script,
  notebook, test e specifiche delle dipendenze;
- `colab_dataset.tar.gz`, contenente immagini, catalogo sorgente, manifest e
  split.

La separazione tra codice e dati permette di aggiornare il software senza dover
ricaricare ogni volta l'intero dataset. Gli archivi sono stati collocati in
Google Drive nella directory:

```text
MyDrive/tesi_qml_colab/colab_bundle/
```

## Inizializzazione e verifica dell'ambiente

Il primo avvio è stato effettuato tramite `colab_quickstart.ipynb`. Il notebook
monta Google Drive, estrae i due archivi nella directory temporanea
`/content/tesi_finale`, installa le dipendenze e registra il progetto Python in
modalità editable. Il dispositivo viene selezionato automaticamente: viene
usato `cuda` quando una GPU è disponibile, altrimenti l'esecuzione prosegue su
CPU.

Poiché il filesystem `/content` è temporaneo, non viene considerato una sorgente
persistente. Il codice e il dataset possono essere riestratti dagli archivi,
mentre risultati e feature vengono conservati separatamente su Drive.

Prima dell'addestramento vengono eseguiti tre livelli di controllo:

1. verifica dello spazio libero e degli inode disponibili;
2. esecuzione della suite di test non quantistica;
3. esecuzione separata dei test dell'implementazione PennyLane.

Durante l'esecuzione reale, la suite non quantistica ha completato 137 test con
esito positivo e ne ha esclusi 9 tramite marker. Questo controllo verifica, fra
gli altri aspetti, configurazioni, split, trasformazioni, inizializzazione dei
modelli, metriche, checkpoint, matrice sperimentale e gestione della sessione
Colab.

### Compatibilità tra PennyLane e JAX

La macchina Colab utilizzata esponeva Python 3.13.15 e una versione recente di
JAX già installata nell'ambiente. PennyLane 0.42.3 non riusciva a importare tale
versione di JAX perché quest'ultima aveva rimosso l'API interna
`DynamicJaxprTrace`. L'errore si manifestava durante la raccolta dei test, prima
dell'esecuzione dell'addestramento.

Per rendere l'ambiente deterministico sono state quindi bloccate esplicitamente
le versioni:

```text
pennylane==0.42.3
jax==0.6.2
jaxlib==0.6.2
```

La correzione è stata inserita sia in `requirements-cluster.txt` sia in
`requirements-colab.txt`, quindi il bundle del codice è stato rigenerato. È
stato inoltre aggiunto un test di regressione che impedisce di produrre nuovi
ambienti Colab privi della coppia JAX/JAXlib compatibile. Dopo il pin, l'intera
suite preliminare è stata completata correttamente.

Questo episodio evidenzia la necessità di bloccare anche le dipendenze già
presenti nell'immagine base di Colab: l'installazione della sola versione di
PennyLane non è sufficiente a garantire la riproducibilità quando una libreria
opzionale incompatibile è preinstallata dal provider.

## Quickstart e primo esperimento

Il quickstart esegue un solo run della matrice. Lo scopo non è ottenere subito
risultati scientifici, ma verificare l'intero percorso operativo:

```text
estrazione → installazione → test → feature → training → valutazione → salvataggio
```

Al termine del primo avvio, lo stato riportava un run completo, nessun fallimento
e 159 run ancora pendenti. La presenza dell'archivio su Drive ha confermato che
il risultato e la feature bank erano stati salvati fuori dalla macchina virtuale
temporanea.

## Calcolo e riuso delle feature

Tutti i modelli ricevono feature estratte dalla stessa ResNet18 preaddestrata con
pesi `IMAGENET1K_V1`. La backbone è congelata e mantenuta in modalità di
valutazione; il vettore prodotto per ogni immagine ha dimensione 512.

Per il training viene generata una sola vista aumentata per coppia di repeat seed
e ObjID. Le viste di validation e test sono invece deterministiche. Le feature
sono memorizzate in una feature bank organizzata per seed, iterazione e split.
Quando la pipeline incontra un run appartenente a un seed già elaborato,
riutilizza la feature bank esistente anziché eseguire nuovamente la ResNet18.

Il task binario non possiede una copia separata delle feature: viene ottenuto
filtrando in memoria la vista triclasse ed escludendo i merger. Questo garantisce
che i campioni condivisi dai due task derivino dagli stessi dati e dalla stessa
procedura di estrazione.

## Addestramento riprendibile

Dopo il quickstart, la campagna viene gestita da `colab_runner.ipynb`. Il
parametro `SESSION_RUN_LIMIT` stabilisce il numero massimo di esperimenti da
tentare nella sessione corrente. Per l'esecuzione complessiva è stato impostato
al numero di run ancora pendenti.

Il notebook delega l'orchestrazione a `scripts/colab_session.py`, che per ogni
sessione:

1. ripristina da Drive lo stato precedente;
2. legge `results/matrix/primary.json`;
3. salta i run già completi;
4. calcola la feature bank richiesta, se ancora assente;
5. avvia un singolo addestramento;
6. salva immediatamente risultati e feature su Drive;
7. passa al run successivo.

Il salvataggio dopo ogni tentativo limita la perdita massima dovuta a una
disconnessione di Colab. Se la sessione viene interrotta durante un run, il
checkpoint `last.pt` consente la ripresa dell'addestramento; i run già completi
non vengono ripetuti.

Ogni modello viene addestrato per un massimo di 80 epoche con AdamW, learning
rate iniziale pari a `1e-3`, weight decay `1e-4`, gradient clipping a 1.0 e
`ReduceLROnPlateau`. L'early stopping usa la validation loss con patience pari a
12. Il checkpoint migliore è selezionato esclusivamente sul validation set; il
test set viene valutato soltanto dopo il ripristino di `best.pt`.

Prima dell'avvio della sessione lunga erano stati completati 5 run senza
fallimenti, mentre 155 risultavano pendenti. Questo valore rappresenta uno stato
operativo intermedio e dovrà essere sostituito, nella versione definitiva della
tesi, con il conteggio finale della matrice.

## Persistenza atomica su Google Drive

Lo stato persistente è memorizzato nel file:

```text
MyDrive/tesi_qml_colab/colab_bundle/colab_state.tar.gz
```

L'archivio contiene due componenti principali:

- `results/`, con stato della matrice, configurazioni, checkpoint, cronologie,
  metriche e predizioni;
- `data/features/`, con le feature bank già calcolate e la relativa provenance.

La scrittura è atomica: il nuovo archivio viene prima costruito come file
temporaneo e sostituisce quello precedente soltanto al termine della creazione.
I file parziali con suffisso `.part` non vengono inclusi. In questo modo, una
disconnessione durante il salvataggio non dovrebbe trasformare un archivio
valido in uno stato incompleto.

Il ripristino accetta soltanto membri appartenenti a `results/` e
`data/features/` e rifiuta percorsi assoluti, link e tentativi di risalita con
`..`. Questi controlli impediscono a un archivio malformato di scrivere al di
fuori della directory del progetto.

## Artefatti prodotti da ogni run

Ogni run completo produce una directory identificata da task, modello, seed,
iterazione e fingerprint della configurazione. Gli artefatti principali sono:

- configurazione completa del run e informazioni sull'ambiente;
- checkpoint migliore e ultimo checkpoint riprendibile;
- cronologia delle metriche per epoca;
- metriche e predizioni di validation;
- metriche e predizioni finali sul test set;
- informazioni su tempi, memoria e numero di parametri;
- checksum delle inizializzazioni e provenance delle feature.

Un run viene marcato come `complete` soltanto quando tutti gli artefatti
obbligatori sono presenti. In caso contrario rimane riprendibile oppure viene
marcato come `failed`, conservando l'informazione sull'errore.

## Aggregazione e produzione delle figure

Al raggiungimento di 160 run completi e zero fallimenti, la cella finale del
runner esegue tre operazioni sequenziali:

1. `aggregate_results.py` raccoglie i risultati individuali e costruisce le
   tabelle aggregate;
2. `generate_figures.py` genera le figure finali della tesi;
3. `colab_session.py save` aggiorna per l'ultima volta l'archivio persistente.

L'aggregazione usa il repeat seed come unità inferenziale. La famiglia primaria
comprende i quattro confronti appaiati Q−B2 sulla macro-F1, distinti per task e
numero di qubit. Per ciascun confronto vengono prodotti differenza media,
deviazione standard, intervallo bootstrap al 95%, permutation test a inversione
di segno, p-value grezzo e corretto con Holm, Cohen's dz e test di Wilcoxon.

Le analisi Q−QF e le metriche specifiche della classe merger sono considerate
secondarie o esplorative. Questa separazione evita di confondere le ipotesi
primarie preregistrate con confronti aggiuntivi interpretati a posteriori.

Una copia finale dell'archivio viene conservata con un nome distinto, ad esempio
`colab_state_complete.tar.gz`, e scaricata anche in locale. La copia rappresenta
lo snapshot riproducibile della campagna e contiene sia gli output necessari
alla tesi sia le feature utili a rieseguire analisi senza ripetere l'estrazione
con ResNet18.

## Considerazioni operative e limiti

Google Colab fornisce un ambiente pratico per questa campagna, ma non garantisce
la durata della sessione, la disponibilità continua della GPU o l'immutabilità
dell'immagine software. La pipeline non presume quindi che i 160 run vengano
completati nella stessa macchina virtuale. La correttezza operativa dipende
invece da tre proprietà verificabili: identificazione univoca dei run, salvataggio
dopo ogni tentativo e ripristino con controllo degli artefatti.

La GPU accelera soprattutto l'estrazione delle feature e le parti tensoriali
compatibili; non implica che la simulazione del circuito quantistico ottenga un
vantaggio hardware quantistico. Tutti i circuiti sono eseguiti tramite
simulazione PennyLane e i risultati non costituiscono una dimostrazione di
vantaggio computazionale o di *quantum advantage*.

Infine, la versione definitiva della tesi dovrà riportare il conteggio finale dei
run, la data di completamento, l'ambiente effettivo registrato negli artefatti e
la dimensione dello snapshot finale. Tali informazioni devono essere ricavate
dai file persistiti, evitando stime basate sull'output temporaneo del notebook.
