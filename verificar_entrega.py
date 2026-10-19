from pathlib import Path
import csv,hashlib,json,zipfile
p=Path(__file__).resolve().parent
manifest=json.loads((p/'SHA256SUMS.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
with zipfile.ZipFile(p/'codigo_congelado.zip') as z:
    for name in z.namelist():
        assert (p/'fuentes'/name).read_bytes()==z.read(name),name
    plan=json.loads(z.read('analisis/particiones_breves_cv.json'))
    people=[]
    for fold in plan['folds']:
        a,b,c=map(set,(fold['train'],fold['validation'],fold['test']))
        assert (len(a),len(b),len(c))==(12,4,4) and not (a&b or a&c or b&c)
        people.extend(c)
    assert len(people)==len(set(people))==20
with (p/'resultados/breves_cv_eventos_prueba.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
assert len(r)==168 and sum(x['detected_overlap']=='True' for x in r)==114
for low,high,count,detected in [(0,10,10,2),(10,30,45,16),(30,float('inf'),113,96)]:
    s=[x for x in r if low<float(x['duration_seconds'])<=high]
    assert len(s)==count and sum(x['detected_overlap']=='True' for x in s)==detected
print('Huellas, cinco divisiones, 168 crisis y recuentos de duración correctos. No se entrenó EEG.')
