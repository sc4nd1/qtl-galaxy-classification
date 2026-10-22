"""Build an HTML gallery and machine-readable index; no external assets."""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
latest = {}
for path in sorted((HERE/'outputs').glob('*/audit.json')):
    audit=json.loads(path.read_text(encoding='utf-8'))
    if audit['status']!='complete':
        continue
    config=audit['settings']
    key=(config['task'],config['subset'],config['qubits'],config['seed'])
    latest[key]=(path,audit)
sections=[]
index=[]
for key,(path,audit) in sorted(latest.items()):
    task,subset,qubits,seed=key
    title=f'{task} · {subset} · {qubits} dimensioni/qubit · seed {seed}'
    sections.append(f'<section><h2>{html.escape(title)}</h2>')
    for example in audit['examples']:
        paths={suffix:(path.parent/'figures'/Path(filename).name).relative_to(HERE).as_posix()
               for suffix,filename in example['figures'].items()}
        error=max(x['max_probability_error'] for x in example['models'].values())
        sections.append(f'<article><a href="{html.escape(paths["pdf"])}">Apri PDF</a>'
                        f'<span> · scarto massimo probabilità: {error:.2g}</span>'
                        f'<img loading="lazy" src="{html.escape(paths["png"])}" '
                        f'alt="Grad-CAM {html.escape(example["true_class"])} {example["objid"]}"></article>')
    sections.append('</section>')
    index.append({'task':task,'subset':subset,'qubits':qubits,'seed':seed,
                  'audit':path.relative_to(HERE).as_posix(), 'examples':len(audit['examples'])})
page='''<!doctype html><html lang="it"><meta charset="utf-8">
<title>Heatmap Grad-CAM — esempi verificati</title>
<style>body{font:17px system-ui;margin:2rem auto;max-width:1400px;padding:0 1rem;color:#172033;background:#f4f5f7}
article{background:white;padding:1rem;margin:1rem 0;border-radius:8px}img{width:100%;height:auto}a{color:#1e3a8a}
p{max-width:1000px;line-height:1.5}h2{margin-top:2.5rem}span{color:#4b5563}</style>
<h1>Grad-CAM: B2, Q e QF</h1>
<p>Esempi reali con predizioni ricostruite verificate contro gli artefatti.
Target comune: classe vera. Risoluzione nativa 7×7; non è una spiegazione causale.
I casi di disaccordo sono etichettati, non sono campioni rappresentativi.
La scala è condivisa entro un'immagine, non tra immagini.</p>'''+''.join(sections)+'</html>'
(HERE/'preview.html').write_text(page,encoding='utf-8')
(HERE/'gallery_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
print(f'{len(index)} analyses, {sum(x["examples"] for x in index)} examples: {HERE/"preview.html"}')
