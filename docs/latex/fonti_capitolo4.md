# Fonti per il capitolo 4 e materiale per i capitoli successivi

Aggiornamento bibliografico per i capitoli 4–5 del 7 settembre 2026:
vedere anche `bibliografia_capitoli4_5.md` per le sei voci aggiunte,
la documentazione versionata e i relativi punti di citazione.

## Uso delle fonti

Il capitolo descrive principalmente una pipeline implementata: formule, parametri
e procedure derivano dal codice e dagli artefatti, non vengono attribuiti
genericamente agli articoli. I lavori sono citati per il paradigma metodologico
o per il concetto specifico cui si riferiscono.

### Fonti locali citate nel capitolo 4

- **Otgonbaatar et al. (2023)** — Quantum Transfer Learning for Real-World, Small,
  and High-Dimensional Remotely Sensed Datasets.
  File: Quantum_Transfer_Learning_for_Real-World_Small_and_High-Dimensional_Remotely_Sensed_Datasets.pdf.
  Esaminati abstract e sezioni di architettura (pagine PDF 1, 3–4).
  Uso: motivazione del transfer learning su dati ad alta dimensionalità e piccoli
  circuiti. Non viene rivendicata una replica dell'ansatz o della misura di
  local effective dimension.
- **Kim, Huh e Park (2023)** — Classical-to-quantum convolutional neural network
  transfer learning. File: 1-s2.0-S092523122300766X-main.pdf.
  Esaminati apertura e struttura QCNN (pagina PDF 3).
  Uso: distinguere la testa variazionale della tesi da una QCNN con pooling.
- **McClean et al. (2018)** — Barren plateaus in quantum neural network training
  landscapes. File: s41467-018-07090-4.pdf.
  Esaminati abstract e introduzione.
  Uso: qualificare il significato del problema dei gradienti; non dichiarare
  automaticamente presenza o assenza di barren plateau nelle run.

### Fonti primarie aggiunte e verificate online

- [Mari et al., Quantum 4, 340 (2020)](https://quantum-journal.org/papers/q-2020-10-09-340/):
  paradigma ibrido e dressed quantum circuit.
- [Loshchilov e Hutter, AdamW](https://arxiv.org/abs/1711.05101):
  weight decay disaccoppiato, pubblicazione ICLR 2019.
- [Bergholm et al., PennyLane](https://arxiv.org/abs/1811.04968):
  differenziazione automatica ibrida.
- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html):
  calibrazione delle reti.
- [Efron, Annals of Statistics 7(1), 1–26 (1979)](https://doi.org/10.1214/aos/1176344552):
  bootstrap; consultata anche una copia del lavoro presso l'Università di Helsinki.
- [Holm, Scandinavian Journal of Statistics 6(2), 65–70 (1979)](https://www.jstor.org/stable/4615733):
  correzione step-down; consultata copia del lavoro presso l'Università di São Paulo.
- [Documentazione AdamW, PyTorch 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html):
  verifica dei default beta/epsilon; valori non sovrascritti nel codice del progetto.

## Materiale locale inventariato, non ancora oggetto di revisione integrale

Non tutti i PDF sono stati letti integralmente. L'inventario di titolo,
metadati/apertura e abstract serve a selezionare quelli pertinenti, senza
riempire la bibliografia di citazioni non utilizzate.

| File | Argomento / destinazione possibile |
|---|---|
| Schuld M. Machine Learning with Quantum Computers 2021.pdf | Schuld e Petruccione, seconda edizione; fondamenti nel capitolo 2. Metadati verificati alle pagine PDF 4–5; DOI 10.1007/978-3-030-83098-4 |
| Shabani J. First Step to Quantum Computing. A Practical Guide for Beginners 2024.pdf | Testo introduttivo; supporto per il capitolo 2 |
| s42484-022-00062-4.pdf | Azevedo, Silva, Dutra: QTL per mammografie; lavori correlati |
| s42484-024-00141-8.pdf | Buonaiuto et al.: QTL per giudizi di accettabilità linguistica; transfer learning fuori dall'imaging |
| 2311.10375v1.pdf | Rath e Date: confronto di encoding; capitolo 2, distinguere gli encoding dal protocollo effettivo |
| 1-s2.0-S1687850726000087-main.pdf | Alfajri et al.: classificazione radiografie multiclass; possibile confronto critico nella discussione |
| 1-s2.0-S003132032500634X-main.pdf | Shu: sparse coding quantistico e capsule; architettura distante dal confronto B2–Q |
| 2602.00128v1.pdf | Akpinar e Oduncuoglu: parallelismo di modelli su MRI; preprint, da valutare per pertinenza |
| computers-15-00065.pdf | Campanile et al.: architetture ibride Quantum–Kolmogorov; lavori correlati |
| aci-10-2025-0448en.pdf | Rawas e AlSaeed: QML per immagini mediche; lavori correlati |
| JTSE_38.pdf | Mandava e Sravanthi: classificazione QML; da esaminare criticamente prima di usarne conclusioni generali |
| 978-3-032-07267-2.pdf | Volume collettivo Next-Gen Healthcare; individuare un capitolo specifico prima di citarlo |
| tesi.pdf | Tesi magistrale di Antonio Tudisco, 2022, sugli encoding; fonte secondaria, non testo da riprodurre |
| FormalizzazioneModelli.pdf | Appunti interni storici, versione architetturale diversa: vedere note_capitolo4.md |

Nessun testo o figura degli articoli è stato copiato nel progetto LaTeX.
I due schemi del capitolo 4 sono originali e rappresentano il codice finale.
