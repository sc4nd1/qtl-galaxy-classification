"""Generate the self-contained Colab notebook; writes only inside heat_map."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE/'gradcam_analysis.py').read_text(encoding='utf-8')
requirements = (HERE/'requirements-colab.txt').read_text(encoding='utf-8')
cells = []


def markdown(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':text.strip().splitlines(keepends=True)})


def code(text):
    text=text.strip()+'\n'
    ast.parse(text)
    cells.append({'cell_type':'code','execution_count':None,'outputs':[],
                  'metadata':{},'source':text.splitlines(keepends=True)})


markdown('''
# Grad-CAM — confronto B2 / Q / QF

Analisi separata dalla tesi e dal training. Nessun file degli esperimenti viene modificato.
Si leggono un archivio di stato e quello del dataset; si estraggono soltanto tre
checkpoint e poche immagini. **Nessun nuovo addestramento e nessun download di galassie.**

La mappa spiega il **logit della classe vera**, mantenendo lo stesso target per tutti
i modelli, anche se le predizioni sono diverse. Il circuito di Q/QF resta nel percorso
del gradiente. Le mappe 7×7 sono interpolate sul crop 224×224: non individuano strutture
con precisione al pixel e non dimostrano un meccanismo causale.

CPU sufficiente per pochi esempi. La prima installazione richiede il download dei
pacchetti e, se assenti, dei pesi ImageNet ResNet18. Tutto l'ambiente scientifico è
isolato in `/content/heat_map/.venv`: non si modifica NumPy nel kernel di Colab,
non serve riavviare il runtime, non si installa JAX.
''')
markdown('''
## 1. Percorsi e selezione

Controlla i due percorsi degli archivi. Per `full` usa `colab_state.tar.gz`;
per 100/300/900 seleziona il tuo `colab_subset_state.tar.gz` e cambia `SUBSET`.
Sono accettate anche directory già estratte, purché contengano le run o il dataset.

Il seed 7 è fissato come esempio, non selezionato come migliore. Si sceglie un'immagine
per classe con una regola deterministica basata sull'ObjID, indipendente dalle predizioni.
L'esempio aggiuntivo di disaccordo è deliberatamente selezionato e viene etichettato.
''')
code('''
from pathlib import Path
import os, sys, subprocess, json, shutil
from datetime import datetime, timezone

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    WORK_DIR = Path('/content/heat_map')
    BUNDLE_DIR = Path('/content/drive/MyDrive/tesi_qml_colab/colab_bundle')
    STATE_SOURCE = BUNDLE_DIR / 'colab_state.tar.gz'
    DATASET_SOURCE = BUNDLE_DIR / 'colab_dataset.tar.gz'
    SAVE_DIR = Path('/content/drive/MyDrive/tesi_qml_colab/heat_map')
else:
    WORK_DIR = Path.cwd() if Path.cwd().name == 'heat_map' else Path.cwd() / 'heat_map'
    PROJECT = WORK_DIR.parent
    STATE_SOURCE = PROJECT / 'colab_pacchetto_finale/colab_state'
    DATASET_SOURCE = PROJECT / 'colab_pacchetto_finale/stack_and_notebooks/colab_dataset.tar.gz'
    SAVE_DIR = WORK_DIR / 'exports'

TASK = 'three_class'       # 'binary' oppure 'three_class'
SUBSET = 'full'            # '100', '300', '900', 'full'
REPEAT_SEED = 7
QUBITS = 4                # 4 oppure 8
PER_CLASS = 1
EXTRA_DISAGREEMENT = True
SELECTION_SEED = 20260907
MANUAL_OBJIDS = []         # opzionale: stringhe ObjID presenti nel TEST del seed
TRUST_MY_CHECKPOINTS = True  # solo archivi tuoi: torch.load usa pickle
WEIGHTS_PATH = None        # opzionale: resnet18-f37072fd.pth già presente
USE_ISOLATED_ENV = IN_COLAB

WORK_DIR = WORK_DIR.resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
for path in (STATE_SOURCE, DATASET_SOURCE):
    if not path.exists():
        raise FileNotFoundError(f'Correggi il percorso: {path}')
print({'work': str(WORK_DIR), 'state': str(STATE_SOURCE), 'dataset': str(DATASET_SOURCE),
       'task': TASK, 'subset': SUBSET, 'qubits': QUBITS, 'seed': REPEAT_SEED})
''')
markdown('''
## 2. Motore incluso nel notebook

Questa cella salva una copia autonoma del motore **dentro heat_map**, con nome distinto
dal sorgente locale. Non modifica o importa i moduli di `tesi_qml`.
''')
code('HELPER_SOURCE = '+repr(source)+'\n\nREQUIREMENTS = '+repr(requirements)+'''
ENGINE = WORK_DIR / 'gradcam_colab_engine.py'
ENGINE.write_text(HELPER_SOURCE, encoding='utf-8')
REQ_FILE = WORK_DIR / 'requirements_colab_generated.txt'
REQ_FILE.write_text(REQUIREMENTS, encoding='utf-8')
print('Motore salvato in', ENGINE)
''')
markdown('''
## 3. Ambiente isolato

Non eseguire installazioni nel runtime principale. L'installazione seguente usa una
virtualenv **senza accesso ai pacchetti globali**, così JAX e i plugin preinstallati di
Colab non interferiscono. Alla prima esecuzione serve connettività per i pacchetti.
Il runtime Colab non è stato eseguito da questo notebook durante la preparazione locale.
''')
code('''
import venv
ENV = dict(os.environ)
for name, folder in [('MPLCONFIGDIR','cache/matplotlib'), ('TORCH_HOME','cache/torch'),
                     ('XDG_CACHE_HOME','cache'), ('TMPDIR','tmp'), ('TEMP','tmp'), ('TMP','tmp')]:
    destination = WORK_DIR / folder
    destination.mkdir(parents=True, exist_ok=True)
    ENV[name] = str(destination)
ENV['PYTHONDONTWRITEBYTECODE'] = '1'
ENV['MPLBACKEND'] = 'Agg'

def live(command):
    print('$', ' '.join(map(str, command)), flush=True)
    with subprocess.Popen(list(map(str, command)), cwd=WORK_DIR, env=ENV,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)

if USE_ISOLATED_ENV:
    env_dir = WORK_DIR / '.venv'
    PYTHON = env_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not PYTHON.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env_dir)
    pip = [PYTHON, '-m', 'pip', 'install', '--cache-dir', WORK_DIR / 'cache/pip']
    live(pip + ['torch==2.8.0', 'torchvision==0.23.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'])
    live(pip + ['-r', REQ_FILE])
else:
    PYTHON = Path(sys.executable)
    print('Uso ambiente locale esistente, senza installare o modificare pacchetti.')

live([PYTHON, '-B', '-c',
      "import torch,torchvision,numpy,pandas,pennylane; "
      "print({'torch':torch.__version__,'torchvision':torchvision.__version__,"
      "'numpy':numpy.__version__,'pandas':pandas.__version__,'pennylane':pennylane.__version__})"])
''')
markdown('''
## 4. Generazione e salvataggio

Prima di accettare una figura vengono verificati: identità delle run, checkpoint,
test condiviso, SHA-256 dell'immagine, probabilità e classe predetta rispetto alle
predizioni archiviate. Tolleranza sulle probabilità: 0,0001; sui logit: 0,001.
**Un disallineamento interrompe l'analisi: non allargare la tolleranza per aggirarlo.**

Ogni esecuzione crea una cartella con timestamp. Dopo il processo viene salvato uno
ZIP in `SAVE_DIR`, anche in caso di errore (con stato `failed`, non pronto per la tesi).
I file `.pt` sono checkpoint propri e fidati: non usare archivi di provenienza sconosciuta.
''')
code('''
if not TRUST_MY_CHECKPOINTS:
    raise RuntimeError('Conferma di usare soltanto i tuoi checkpoint fidati.')

command = [PYTHON, '-B', '-u', ENGINE,
           '--state', STATE_SOURCE, '--dataset', DATASET_SOURCE,
           '--task', TASK, '--subset', SUBSET, '--seed', REPEAT_SEED,
           '--qubits', QUBITS, '--per-class', PER_CLASS,
           '--selection-seed', SELECTION_SEED, '--device', 'cpu', '--trust-checkpoints']
if EXTRA_DISAGREEMENT:
    command += ['--extra-disagreement']
if MANUAL_OBJIDS:
    command += ['--objids', ','.join(map(str, MANUAL_OBJIDS))]
if WEIGHTS_PATH:
    command += ['--weights', WEIGHTS_PATH]

before = set((WORK_DIR / 'outputs').glob('*')) if (WORK_DIR / 'outputs').exists() else set()
LAST_OUTPUT = None
try:
    live(command)
finally:
    generated = set((WORK_DIR / 'outputs').glob('*')) - before
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for folder in sorted(generated):
        LAST_OUTPUT = folder
        archive = shutil.make_archive(str(SAVE_DIR / folder.name), 'zip', folder)
        print('Salvato:', archive)

if LAST_OUTPUT is None:
    raise RuntimeError('Il processo non ha prodotto una cartella di analisi.')
AUDIT = json.loads((LAST_OUTPUT / 'audit.json').read_text(encoding='utf-8'))
if AUDIT['status'] != 'complete':
    raise RuntimeError('Analisi non completa: leggere audit.json, non usare le figure nella tesi.')
print({'status': AUDIT['status'], 'examples': len(AUDIT['examples']), 'output': str(LAST_OUTPUT)})
''')
markdown('''
## 5. Anteprima e lettura

Ogni figura confronta B2, Q e QF sullo **stesso target**. Per un errore non stiamo
automaticamente spiegando la classe sbagliata predetta: spieghiamo sempre quella vera.
La barra usa il massimo positivo comune ai tre pannelli della stessa immagine;
non confrontare l'intensità fra immagini né interpretarla come probabilità o importanza
causale. Una mappa vuota significa assenza di contributi positivi secondo questa regola,
non assenza generale di informazione. I valori grezzi, anche firmati, sono salvati nei `.npz`.

Per cambiare 4→8 qubit basta modificare `QUBITS` e ripetere la cella di generazione.
Per i subset cambiare anche archivio e `SUBSET`. Nessun training viene riavviato.
''')
code('''
from IPython.display import display, Image as DisplayImage
if AUDIT['status'] != 'complete':
    raise RuntimeError('Mostrare soltanto analisi complete.')
for path in sorted((LAST_OUTPUT / 'figures').glob('*.png')):
    display(DisplayImage(filename=str(path), width=1200))
''')
markdown('''
## Uso nella tesi

- Capitolo 4: metodo Grad-CAM e un esempio esplicitamente illustrativo.
- Capitolo 5: confronto qualitativo, senza inferire superiorità globale da un seed.
- Non aggiornare automaticamente i capitoli o la bibliografia principale.
- PDF per Overleaf in `figures/`; `audit.json` e `selected_examples.csv` conservano
  la provenienza; `maps/` conserva le mappe grezze e i gradienti.

Riferimento: Selvaraju et al., *Grad-CAM: Visual Explanations From Deep Networks
via Gradient-Based Localization*, ICCV 2017, pp. 618–626.
[Articolo originale](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                  'language_info':{'name':'python'},'colab':{'name':'07_colab_gradcam.ipynb'}},
          'nbformat':4,'nbformat_minor':5}
for i,cell in enumerate(cells):
    cell['id']=f'heatmap-{i:02d}'
path=HERE/'07_colab_gradcam.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(path)
