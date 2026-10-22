# Bibliografia dei capitoli 4 e 5

Aggiornamento: 7 settembre 2026, successivo alla stesura del capitolo 5.

La bibliografia condivisa è `bibliografia.bib`, gestita con biblatex/Biber.
Comprende ora 20 voci: conservate le 14 esistenti e aggiunte sei fonti
pertinenti ai metodi effettivamente descritti. Nessuna voce è stata
aggiunta soltanto perché il relativo PDF era presente in `docs`.

## Sei aggiunte e relativi impieghi

| Chiave BibTeX | Capitolo / impiego | Riscontro primario |
|---|---|---|
| `paszke2019pytorch` | 4, framework PyTorch; distinta dalla citazione di PennyLane | [Atti NeurIPS 2019, volume 32](https://papers.nips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) |
| `schuld2019gradients` | 4, parameter-shift come alternativa, esplicitamente **non** usata nelle run finali | [Physical Review A 99(3), 032331 (2019)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331) |
| `sklearn17f1` | 4, Macro-F1 come media non pesata delle F1 per classe, non media armonica di due medie macro | [Documentazione scikit-learn 1.7.2](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.f1_score.html) |
| `sklearn17averageprecision` | 4, distinzione tra average precision e integrazione trapezoidale della curva PR | [Documentazione scikit-learn 1.7.2](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html) |
| `gneiting2007scoring` | 4 e 5, qualità delle distribuzioni probabilistiche; Brier e log-loss non sono errori di sola calibrazione | [Copia dell'articolo presso l'università dell'autore](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf), JASA 102(477), 359–378, DOI 10.1198/016214506000001437 |
| `nadeau2003inference` | 4 e 5, cautela sull'incertezza nei confronti con training/test riutilizzati | [Machine Learning 52, 239–281 (2003)](https://link.springer.com/article/10.1023/A:1024068626366) |

Consultati i metadati/abstract delle pagine primarie e i passaggi pertinenti
dei documenti; per Gneiting e Raftery anche introduzione e §3, esempi Brier
e logarithmic score. Non si dichiara una lettura integrale di tutti gli
articoli. La pagina editoriale di Gneiting e Raftery non era accessibile;
è stata usata la copia universitaria del lavoro pubblicato.

La citazione di Nadeau e Bengio **non** implica che sia stato applicato
il loro stimatore corretto: nel capitolo 4 si precisa esplicitamente che
non è quello usato. Gli IC e i p-value dell'esperimento non sono stati
modificati con l'aggiornamento bibliografico.

## Fonti già presenti richiamate nel capitolo 5

- `efron1979bootstrap`: principio del ricampionamento; numero delle repliche
  e specifico intervallo percentile derivano dall'implementazione documentata.
- `holm1979multiple`: correzione delle due famiglie Q−B2, separate.
- `guo2017calibration`: lettura dell'ECE e distinzione tra confidenza e accuratezza.

Il capitolo 5 cita inoltre le nuove voci `nadeau2003inference` e
`gneiting2007scoring`. Le altre fonti teoriche restano nel capitolo 4,
senza ripeterle accanto a ogni figura dei risultati.

## Versioni software e cautele

- La voce `torchvisionresnet18` ora punta alla documentazione **0.23**, coerente
  con l'ambiente registrato, anziché al ramo mobile `main`. URL verificato
  e data di consultazione aggiornata al 7 settembre 2026.
- Le nuove pagine scikit-learn sono versionate **1.7** (documentazione 1.7.2),
  coerenti con le run; sono fonti dell'API, non articoli sperimentali.
- Le versioni precise di PyTorch e PennyLane restano quelle negli artefatti;
  l'anno degli articoli dei framework non va confuso con la versione installata.
- Gli esiti negativi, i tempi e le metriche della tesi sono risultati originali
  dell'esperimento: non vengono attribuiti alla letteratura esterna.
- Non si aggiungono DOI, numeri di pagina o autori non verificati. Il preprint
  PennyLane conserva l'identificativo arXiv esistente.

## Verifica LaTeX

Controllare che tutte le chiavi citate siano presenti una sola volta nel
BibTeX. Per Overleaf usare pdfLaTeX con Biber e ricompilare il documento
scelto. La numerazione è automatica in ordine di citazione: può cambiare
inserendo nuovi capitoli, senza richiedere correzioni manuali nel testo.
