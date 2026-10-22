# Tesi adattata al template del relatore

Documento principale: `main.tex`. Compilatore: **pdfLaTeX**.
Bibliografia: **BibTeX**, stile `plain` del template (non Biber).
I listati usano **minted**, stile **manni**, come negli esempi del professore.
Overleaf supporta minted; un'eventuale compilazione locale richiede anche
gli strumenti esterni previsti dalla versione di minted installata.

## Cosa è stato conservato

- Classe `report`, formato A4, fronte-retro e apertura dei capitoli a destra.
- Margini 2,5 cm sopra/sotto e 3 cm ai lati, carattere da 12 punti.
- Testatine, numerazione e stile delle pagine iniziali dei capitoli.
- Struttura e logo del frontespizio; nome del relatore.
- Dichiarazione di responsabilità, invariata.
- Bibliografia `plain` e strumenti per annotazioni del relatore.
- Colori `manni` dei listati; impostazioni del template applicate a Python e SQL.

## Adattamenti circoscritti

- Corretta la `c` accidentale prima del primo commento in `main.tex`.
- Inseriti titolo, candidato, matricola e anno accademico della tesi revisionata.
  Il titolo usa `LARGE` anziché `huge` per contenerne la lunghezza.
- Sostituito il testo dimostrativo con i sette capitoli e l'appendice SQL.
  Capitoli e figure sono identici alla versione revisionata, salvo gli involucri
  dei listati convertiti a minted.
- Sostituito il segnaposto dell'abstract con un abstract inglese coerente con
  metodo e risultati. Da rileggere e approvare insieme al testo finale.
- Aggiunti soltanto i pacchetti e comandi richiesti dalle formule, tabelle,
  schemi TikZ e riferimenti della tesi, senza caricare il vecchio preambolo.
- Ripristinato il numero nelle didascalie delle tabelle: necessario per
  identificare i richiami numerati nel testo.
- Conservato l'indice sintetico a capitoli e sezioni.
- Convertite le 23 voci bibliografiche al formato compatibile con BibTeX plain:
  `journal`, numeri articolo nel campo `pages`, risorse web come `misc` e link
  DOI/URL nelle note. Nessuna fonte aggiunta o rimossa.
- Disattivati citazione personale e ringraziamenti ancora dimostrativi;
  eliminato dal documento l'esercizio di logica di esempio.
- Rimossi i richiami manuali a pagine vuote tra contenuti preliminari;
  `openright` rimane attivo, quindi sono possibili pagine di compensazione
  previste dal template. Non è stata trasferita la vecchia forzatura `openany`.

## Caricamento

Importare `tesi_template_relatore_overleaf.zip` come nuovo progetto Overleaf,
oppure caricare il suo contenuto nel progetto del template. Impostare `main.tex`
come documento principale. Non mescolare `preambolo.tex` o `TemplateTesi1.tex`
del vecchio progetto con questo main.

Gli esempi originali sono conservati nella cartella locale ma non inclusi
nello ZIP pronto per Overleaf. Per aggiungere ringraziamenti o citazione,
creare o caricare i relativi file in `contents/`, compilarne il contenuto
e attivare le righe commentate in `main.tex`.

## Verifiche e limite

Controllati staticamente percorsi, citazioni, etichette, ambienti, conservazione
dei capitoli e delle figure e sintassi dei quattro snippet Python.
Non è disponibile un compilatore TeX locale: occorre ricompilare su Overleaf
e verificare soprattutto il frontespizio con titolo lungo, la distribuzione
dei listati e i nuovi salti di pagina. Non è stato prodotto un PDF compilato.

Le cartelle originali `2025_Tesi3___Scandone_Michele` e
`tesi_latex_overleaf` non sono state modificate.
