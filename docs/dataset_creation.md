# Creazione del dataset SDSS DR17

Questo documento descrive il dataset rigenerato il 27 agosto 2026 e gli artefatti
effettivamente presenti in `tesi_finale/data`. I conteggi riportati derivano dai
manifest generati dalla pipeline, non dal semplice conteggio dei file nelle cartelle.

## 1. Origine e criteri scientifici

Il catalogo proviene dalla Sloan Digital Sky Survey Data Release 17 (SDSS DR17).
La query CasJobs incrocia `PhotoObj`, `zooSpec` e `SpecObj`:

```sql
SELECT
    p.objid,
    p.ra,
    p.dec,
    z.p_el_debiased AS prob_elliptical,
    z.p_cs_debiased AS prob_spiral,
    z.p_mg AS prob_merger,
    s.z AS redshift
INTO mydb.GalaxyCandidates_Final
FROM PhotoObj AS p
JOIN zooSpec AS z ON p.objid = z.objid
JOIN SpecObj AS s ON p.objid = s.bestObjID
WHERE
    p.clean = 1
    AND p.type = 3
    AND p.g BETWEEN 14 AND 17.5
    AND s.z BETWEEN 0.005 AND 0.1
    AND (z.p_el_debiased > 0.8 OR z.p_cs_debiased > 0.8 OR z.p_mg > 0.4)
```

Le hard label sono assegnate solo quando una riga supera esattamente una delle
seguenti soglie:

| Classe | Condizione |
|---|---|
| `elliptical` | `prob_elliptical > 0.8` |
| `spiral` | `prob_spiral > 0.8` |
| `merger` | `prob_merger > 0.4` |

La pipeline applica inoltre `0.005 < redshift < 0.1` e rifiuta probabilità,
coordinate o valori numerici non validi. Il CSV conservato nel progetto contiene
le colonne selezionate dalla query, ma non contiene `clean`, `type` e `g`: questi
tre vincoli restano documentati dalla query e non possono essere verificati di
nuovo usando soltanto il CSV esportato.

## 2. Integrità degli ObjID

`objid` è sempre letto come stringa. Non deve essere convertito in `float` o
aperto e risalvato con software che rappresenta gli identificativi a 19 cifre in
notazione scientifica. Il nome di ogni immagine è l'ObjID esatto:

```text
data/images/<label>/<objid>.jpg
```

Questa regola corregge il problema della pipeline storica, nella quale il passaggio
attraverso un numero floating-point poteva arrotondare l'identificativo e causare
collisioni tra oggetti o classi.

Il catalogo `GalaxyCandidates.csv` contiene 65.906 righe. La pulizia ha prodotto:

| Passaggio | Righe |
|---|---:|
| Input | 65.906 |
| Duplicati esatti rimossi | 5 |
| Righe escluse per ObjID duplicato e discordante | 72 |
| Candidati puliti | 65.829 |

I 72 record esclusi corrispondono a 36 ObjID conflittuali: vengono scartate tutte
le occorrenze, senza scegliere arbitrariamente una riga.

I candidati puliti sono distribuiti così:

| Classe | Candidati puliti |
|---|---:|
| `elliptical` | 22.163 |
| `merger` | 1.877 |
| `spiral` | 41.789 |

## 3. Pool di download e immagini

La classe minoritaria `merger` viene mantenuta integralmente. Per `elliptical` e
`spiral` viene estratta, con seed 42, una riserva pari a 1,25 volte la numerosità
dei merger. Il pool risultante è:

| Classe | Cutout scaricati |
|---|---:|
| `elliptical` | 2.347 |
| `merger` | 1.877 |
| `spiral` | 2.347 |
| **Totale** | **6.571** |

I cutout provengono dall'endpoint SDSS DR17 `ImgCutout/getjpeg`, con i parametri:

```text
width  = 320 pixel
height = 320 pixel
scale  = 0.396 arcsec/pixel
```

La dimensione 320×320 è intenzionale. Dopo una rotazione arbitraria, il quadrato
inscritto ha lato circa `320 / sqrt(2) = 226` pixel, sufficiente per eseguire un
`CenterCrop(224)` senza introdurre bordi neri nel tensore fornito a ResNet18.

Il download è atomico e riprendibile. Ogni risposta deve essere un'immagine
decodificabile, con dimensione esatta 320×320 e contrasto non degenere. La
validazione calcola inoltre l'hash SHA-256. Nella rigenerazione corrente:

- tutte le 6.571 immagini sono risultate valide;
- non sono state trovate immagini mancanti o illeggibili;
- non sono stati trovati duplicati visuali né inter-classe né intra-classe.

## 4. Dataset sperimentale finale

Dopo la deduplicazione, il bilanciamento usa la numerosità della classe minoritaria
e la arrotonda per difetto al multiplo di quattro più vicino. Da 1.877 si passa
quindi a 1.876 campioni per classe:

| Classe | Dataset finale |
|---|---:|
| `elliptical` | 1.876 |
| `merger` | 1.876 |
| `spiral` | 1.876 |
| **Totale** | **5.628** |

Ogni fold contiene 469 immagini per classe, cioè 1.407 immagini totali. Le 943
immagini valide non selezionate restano documentate in
`manifest_exclusions.parquet` con motivo `class_balance_sampling`.

## 5. Artefatti riproducibili

Gli artefatti che definiscono il dataset sono:

| File | Funzione |
|---|---|
| `candidates.parquet` | catalogo pulito con hard label e ObjID esatti |
| `downloads.parquet` | pool selezionato per l'acquisizione |
| `validated.parquet` | esito dei controlli e hash per ogni cutout |
| `final.parquet` | campione deduplicato e bilanciato usato dagli esperimenti |
| `manifest_summary.json` | riepilogo numerico e motivi di esclusione |
| `splits/all.parquet` | assegnazioni train/validation/test per seed e iterazione |

Gli esperimenti devono leggere `final.parquet` e `splits/all.parquet`; non devono
costruire il campione enumerando le directory.

Le 5.212 immagini della pipeline precedente, tutte 256×256, sono state separate
in `tesi_finale/data_legacy_256_20260827/images` e non fanno parte del dataset
attivo.
