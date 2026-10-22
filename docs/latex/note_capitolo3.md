# Note editoriali e di verifica — capitolo 3

## Fonti interne

- dataset_creation.md: query, conteggi e costruzione del campione.
- configs/dataset.yaml: soglie, dimensioni dei cutout, seed e task.
- src/tesi_qml/data/manifests.py: pulizia e bilanciamento.
- src/tesi_qml/data/validation.py: controlli dei JPEG e hash.
- src/tesi_qml/data/splits.py: partizioni e prima assegnazione.
- src/tesi_qml/data/transforms.py: ordine delle trasformazioni.
- src/tesi_qml/data/features.py: backbone e cache.
- notebooks/colab_quickstart.ipynb: patch runtime dei subset,
  seed condiviso, fingerprint della selezione e audit di annidamento.
- results/figure_dataset/figures: PDF e PNG del dataset.

Attenzione: il notebook applica patch al runtime Colab; il solo scripts/train.py
locale non descrive necessariamente il codice effettivamente usato per i subset.
Non sostituire il protocollo eseguito con i default di una versione precedente.

## Precisazioni introdotte nel testo

1. Repeated holdout: dieci seed e soltanto iteration=1, non quattro-fold CV completa.
2. L'implementazione con quattro blocchi viene spiegata una sola volta per
   riconciliare il codice con il protocollo descritto.
3. Full significa 1.876/2.814 immagini di training, non 5.628.
4. I subset contengono 100/300/900 immagini totali, con validation e test invariati.
5. Gli hash SHA-256 identificano file byte-identici; il testo non ripete la
   formulazione più forte “assenza di duplicati visuali” del documento storico.
6. BETWEEN nella query include gli estremi; il filtro locale sul redshift è stretto.
7. clean/type/g non sono verificabili dal CSV esportato.
8. Le frazioni di voto e le soglie non sono probabilità calibrate di correttezza.
9. Una vista aumentata per oggetto nella cache non equivale ad augmentation
   rinnovata a ogni epoca.
10. Holdout ripetuti sullo stesso dataset non sono campioni astronomici indipendenti.
11. Il bilanciamento delle classi non bilancia redshift o condizioni osservative.

## Figure

Incluse senza alterare gli originali:

- 01_data_attrition.pdf;
- 02_class_balance.pdf;
- 03_vote_fractions.pdf;
- 04_redshift_distribution.pdf.

05_repeated_holdout.pdf non è incluso: nell'anteprima PNG la legenda si
sovrappone al titolo. La tabella delle partizioni comunica già le quote esatte,
senza aggiungere una figura ridondante. Nessuna immagine è stata riscaricata.

I grafici aggregati delle prestazioni restano in results e sono destinati
ai capitoli successivi, non a quello sul dataset.

## Bibliografia

Verificate su fonti primarie le voci SDSS DR17, Galaxy Zoo 1, Darg et al.,
ResNet e documentazione ufficiale TorchVision. Le scelte specifiche della
pipeline sono descritte come scelte dell'esperimento, non attribuite agli articoli.

## Da completare con l'autore

- Corso di laurea, relatore, eventuale correlatore, candidato, matricola,
  anno accademico e logo autorizzato.
- Titolo definitivo, al termine della revisione complessiva.
- Eventuale collocazione di alcune figure descrittive in appendice.
- Revisione scientifica del relatore: questa è una prima bozza.

## Verifica tecnica

Controllo statico previsto per file inclusi, immagini, chiavi bibliografiche,
etichette, ambienti e parentesi graffe. Nessun motore TeX disponibile localmente:
non dichiarare il PDF compilato o l'impaginazione verificata.
