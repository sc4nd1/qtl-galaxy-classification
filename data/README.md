# Dati

Il dataset attivo rigenerato il 27 agosto 2026 contiene 6.571 cutout SDSS DR17
validi a 320×320. Il manifest sperimentale finale seleziona 5.628 immagini
bilanciate: 1.876 per classe e 469 per classe in ciascuno dei quattro fold.

- `source/`: catalogo originale, da conservare immutato.
- `manifests/`: audit, pool download, validazione e manifest finale.
- `images/`: cutout SDSS DR17 320×320, organizzati per hard label.
- `splits/`: assegnazioni train/validation/test condivise da tutti i modelli.
- `features/`: feature bank ResNet18 per repeat seed e iterazione.

Le immagini e le feature sono rigenerabili; manifest e split sono gli artefatti che definiscono il campione sperimentale. Non costruire dataset enumerando le cartelle.

I precedenti cutout preliminari a 256×256 sono stati completamente dismessi e non fanno parte del dataset del progetto. L'unico dataset attivo è costituito dai 6.571 cutout a risoluzione 320×320 pixel.
