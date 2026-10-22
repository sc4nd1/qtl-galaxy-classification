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

Le immagini legacy a 256×256 sono archiviate separatamente in
`../data_legacy_256_20260827/images` e non appartengono al dataset attivo.
