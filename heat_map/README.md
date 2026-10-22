# Heatmap Grad-CAM — analisi separata

Tutto il lavoro nuovo è in questa directory. Non sono stati modificati
capitoli, bibliografia principale, notebook precedenti, dati o checkpoint originali.
Non è stato eseguito nuovo training.

## Da aprire

- **`07_colab_gradcam.ipynb`**: notebook Colab autonomo, include il motore;
  non devi caricare separatamente lo script Python.
- **`preview.html`**: galleria locale degli esempi già generati e verificati.
- **`outputs/`**: figure PDF/PNG, audit, selezione degli oggetti e mappe numeriche.
- `gradcam_analysis.py`: motore utilizzabile anche da terminale.
- `metodo_gradcam.tex` e `references.bib`: proposta di testo e riferimento
  da valutare per il capitolo 4, **non inseriti automaticamente nella tesi**.

## Che cosa rappresentano le figure

Ogni esempio mostra il crop effettivo di test e tre sovrapposizioni:
B2, Q e QF. La classe spiegata è **sempre la classe vera**, anche se
un classificatore ne predice un'altra. La probabilità mostrata è quella
della classe spiegata, non necessariamente la confidenza massima.

La ResNet18 è congelata e identica: non impara nuove feature in questa
analisi. Si studia come le diverse teste utilizzano la sua rappresentazione.
Il gradiente del logit attraversa tutta la testa, incluso il circuito
quantistico nel caso di Q e QF, fino all'output di `layer4`.
Il calcolo usa la media spaziale del gradiente per pesare i 512 canali,
poi somma i canali e applica ReLU (Grad-CAM positivo).

La mappa originale ha risoluzione **7×7**. L'interpolazione bilineare
a 224×224 non aggiunge informazione e non giustifica interpretazioni
precise di nuclei, bracci o singoli pixel. Non è una attention map
nativa, una spiegazione completa o una prova causale del ragionamento.

La scala è condivisa tra B2/Q/QF entro la stessa immagine, dividendo
per il massimo positivo dei tre CAM interpolati. Le intensità non sono
probabilità e dipendono anche dalla scala dei logit/gradienti dei modelli:
non usarle per misurare quale rete “capisce di più”. Le scale di immagini
diverse non sono confrontabili. Le mappe firmate sono conservate nei NPZ,
ma le figure mostrano solo contributi positivi. Un CAM nullo resta nullo.

## Selezione e validazione

Default: task a tre classi, Full, seed 7, 4 qubit, un oggetto per classe.
Gli oggetti sono ordinati con SHA-256 di `selection_seed|objid`; non
vengono selezionati per bellezza della heatmap, correttezza o confidenza.
Un ulteriore esempio di disaccordo può essere aggiunto: è esplicitamente
etichettato come caso selezionato, non come campione rappresentativo.
Cambiare seed o scegliere manualmente gli ObjID è possibile, ma va dichiarato.

Prima di accettare un esempio il motore verifica:

1. Una sola run per combinazione task/subset/seed/dimensione/modello.
2. Provenance e inizializzazioni appaiate; gli stessi oggetti di test.
3. Firma dell'ansatz finale RY → due blocchi RZ/RY + catena CNOT aperta.
4. Fingerprint del checkpoint `best.pt` coerente con `run_config.json`.
5. Pesi ufficiali ResNet18 ImageNet1K V1 e hash dell'immagine originale.
6. Preprocessing effettivo: RGB, center crop 224, normalizzazione ImageNet,
   **senza il resize automatico del preset TorchVision**.
7. Probabilità ricostruite entro 1e-4 da quelle salvate; stessa classe predetta;
   logit entro 1e-3. Un disallineamento fa fallire l'analisi.

Il controllo di predizione non costituisce un test di fedeltà causale della
spiegazione. Non sono stati eseguiti esperimenti di randomizzazione dei pesi
o occlusione: i grafici restano analisi qualitative, non evidenza statistica
di superiorità tra architetture. L'audit viene salvato anche in caso di errore:
usare per la tesi solo cartelle con `"status": "complete"`.

## Colab: esecuzione

1. Apri `07_colab_gradcam.ipynb`, basta un runtime CPU.
2. Controlla i percorsi della prima cella:
   - Full: `.../colab_bundle/colab_state.tar.gz`;
   - subset: `.../colab_bundle/colab_subset_state.tar.gz` (verifica il tuo nome);
   - immagini: `.../colab_bundle/colab_dataset.tar.gz`.
3. Esegui le celle in ordine. Non eseguire i vecchi setup di training.
4. Gli output locali sono in `/content/heat_map/outputs`; al termine vengono
   copiati come ZIP in `MyDrive/tesi_qml_colab/heat_map`.

Il notebook crea un ambiente isolato in `/content/heat_map/.venv`, senza
accesso ai pacchetti globali di Colab. Non aggiorna NumPy nel kernel e
non richiede il riavvio. Non installa JAX né i plugin CUDA JAX, inutili
per l'interfaccia Torch usata qui. Il notebook è validato staticamente;
l'installazione della virtualenv su Colab non è stata eseguita localmente.

La prima esecuzione scarica i pacchetti e, se necessario, i pesi ufficiali
ResNet18 (~45 MB). Non scarica galassie: legge dall'archivio soltanto quelle
selezionate. Gli archivi originali sono aperti in sola lettura.
Il caricamento dei checkpoint usa pickle: impiegare **solo i propri archivi
fidati**, non file `.pt` provenienti da terzi sconosciuti.

## Esecuzione locale

Dalla radice del repository, usando un ambiente con Torch, TorchVision,
PennyLane, NumPy, pandas, Pillow, Matplotlib e supporto Parquet:

```powershell
py -3.10 -B heat_map/gradcam_analysis.py `
  --state colab_pacchetto_finale/colab_state `
  --dataset colab_pacchetto_finale/stack_and_notebooks/colab_dataset.tar.gz `
  --task three_class --subset full --seed 7 --qubits 4 `
  --extra-disagreement --trust-checkpoints
```

Opzionale: `--weights PERCORSO/resnet18-f37072fd.pth` per riusare una copia
locale ufficiale senza scaricarla. Per i subset passare lo snapshot corretto
e `--subset 100` (oppure 300/900). Gli output hanno sempre un timestamp nuovo.

```powershell
py -3.10 -B heat_map/test_heatmaps.py
py -3.10 -B heat_map/build_notebook.py
py -3.10 -B heat_map/build_gallery.py
```

Il motore è stato eseguito sui checkpoint reali in locale. L'ambiente locale
usa Torch 2.10 / TorchVision 0.25 / PennyLane 0.42.3: le differenze rispetto
all'ambiente di training sono registrate in ciascun `audit.json`, insieme
agli scarti effettivi delle predizioni. Non è stata cambiata l'installazione locale.

## Come usarle nella tesi

Nel capitolo 4 descrivere il metodo e mostrare un esempio illustrativo.
Nel capitolo 5 usare eventualmente un confronto corretto e uno di errore,
senza trasformare un singolo seed in prova generale. La classe vera,
quella predetta, il regime, il seed e la dimensione sono dichiarati nelle figure.
Non usare il solo colore per affermare che la rete identifica strutture
fisiche: serve distinguere osservazione visiva e ipotesi interpretativa.

Riferimento: [Selvaraju et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
