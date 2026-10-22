# Modulo di Interpretabilità Visiva (Grad-CAM)

Questo modulo implementa l'analisi di Explainable AI (XAI) tramite **Gradient-weighted Class Activation Mapping (Grad-CAM)** ([Selvaraju et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)) per confrontare l'attribuzione spaziale delle feature visive tra le teste di classificazione classica ($B2$) e quantistica ($Q$, $QF$) applicate alla medesima backbone ResNet-18.

---

## 1. Struttura del Modulo

- [`gradcam_analysis.py`](gradcam_analysis.py): motore di inferenza, retropropagazione dei gradienti e generazione delle mappe Grad-CAM, eseguibile da riga di comando.
- [`07_colab_gradcam.ipynb`](07_colab_gradcam.ipynb): notebook autonomo per l'esecuzione del protocollo su Google Colab con gestione degli archivi di stato e download dei pesi ufficiali.
- [`test_heatmaps.py`](test_heatmaps.py): suite di test di regressione per la verifica delle ricostruzioni probabilistiche, dell'allineamento dei logit e della corretta propagazione dei gradienti quantistici.

---

## 2. Formulazione Matematica e Principio di Funzionamento

La backbone convoluzionale ResNet-18 (`IMAGENET1K_V1`) è mantenuta identica e congelata per tutte le architetture. Lo studio valuta come le diverse teste di classificazione ($B2$, $QF$, $Q$) retropropagano il gradiente sulle feature estratte all'uscita dell'ultimo blocco convoluzionale (`layer4`), caratterizzato da una griglia spaziale $7 \times 7$ con $512$ canali.

### 2.1 Calcolo delle Mappe di Attribuzione

1. **Retropropagazione del Logit**:
   Dato il logit $y^c$ associato alla classe target $c$ (per protocollo scientifico, **sempre la classe reale** dell'oggetto astronomico), il gradiente attraversa l'intera testa (inclusi i gate quantistici differenziabili e le proiezioni lineari) fino alle mappe di attivazione $A^k \in \mathbb{R}^{7 \times 7}$ del canale $k \in \{1, \dots, 512\}$ di `layer4`:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^7 \sum_{j=1}^7 \frac{\partial y^c}{\partial A_{i, j}^k}$$

2. **Combinazione Lineare e Rettificazione**:
   I pesi di importanza $\alpha_k^c$ fungono da coefficienti per la combinazione lineare delle mappe di feature, a cui viene applicata la rettificazione ReLU per isolare unicamente i gradienti a contribuzione positiva:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{512} \alpha_k^c A^k \right) \in \mathbb{R}^{7 \times 7}$$

3. **Interpolazione e Sovrapposizione**:
   La matrice $7 \times 7$ risultante viene interpolata bilinearmente a $224 \times 224$ pixel per consentire la sovrapposizione visiva al crop di test.
   > **Vincolo di risoluzione**: l'interpolazione bilineare serve unicamente alla visualizzazione qualitativa e non aggiunge contenuto informativo: non supporta inferenze a livello di singoli pixel, filamenti o dettagli stellari sub-risolti.

4. **Normalizzazione Congiunta**:
   Entro la medesima immagine di galassia, la mappa termica è normalizzata congiuntamente per il massimo positivo registrato tra i tre modelli confrontati ($B2$, $QF$, $Q$):
   $$\widetilde{L}_m^c = \frac{L_m^c}{\max_{m' \in \{B2, Q, QF\}} \left( \max L_{m'}^c \right)}$$
   Le intensità dipendono dalla scala numerica dei logit e dei gradienti dei singoli modelli: non costituiscono probabilità assolute e non vanno intese come misure quantitative di accuratezza percettiva.

---

## 3. Protocollo di Selezione e Controlli di Integrità

Per garantire la riproducibilità ed escludere selezioni arbitrarie (*cherry-picking*):
- I campioni di test sono estratti in modo deterministico mediante ordinamento pseudocasuale basato su hash SHA-256 (`selection_seed|objid`).
- Prima di accettare una ricostruzione, il motore esegue i seguenti controlli vincolanti:
  1. Corrispondenza univoca della tupla `(task, subset, seed, qubits, model)`.
  2. Verifica delle inizializzazioni appaiate e dell'identità degli indici del test set sigillato.
  3. Verifica del fingerprint del checkpoint `best.pt` con `run_config.json`.
  4. Preprocessing rigoroso: estrazione crop centrale a 224 pixel dall'immagine originale 320×320 e normalizzazione ImageNet, senza resize intermedio.
  5. **Verifica numerica di fedeltà**: le probabilità previste devono corrispondere a quelle registrate in `test_predictions.parquet` entro una tolleranza di $10^{-4}$, con scarto sui logit inferiore a $10^{-3}$. Qualsiasi divergenza superiore provoca l'interruzione con errore.

---

## 4. Esecuzione

### Esecuzione Locale

Dalla radice del repository:

```powershell
# Esecuzione dell'analisi Grad-CAM su task triclasse, repeat seed 7, 4 qubit
python heat_map/gradcam_analysis.py `
  --state results `
  --dataset data/images `
  --task three_class --subset full --seed 7 --qubits 4 `
  --trust-checkpoints

# Esecuzione della suite di test del modulo Grad-CAM
pytest heat_map/test_heatmaps.py -q
```

### Esecuzione su Google Colab

1. Aprire il notebook [`07_colab_gradcam.ipynb`](07_colab_gradcam.ipynb) in un runtime Colab (GPU o CPU).
2. Verificare i percorsi dei bundle in Google Drive (`colab_state.tar.gz` o `colab_subset_state.tar.gz` e `colab_dataset.tar.gz`).
3. Eseguire sequenzialmente le celle del notebook per generare i grafici e archiviare l'output in `MyDrive/tesi_qml_colab/heat_map/`.
