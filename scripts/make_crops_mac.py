"""
Run on your Mac: downloads RUKOPYS, cuts text-line crops, and packs them into crops.tar.
Upload crops.tar to Google Drive -> MyDrive/rukopys_trocr/crops.tar, then in the Colab
notebook run cells 1-3: cell 3 unpacks it and you can go straight to training (cell 6).

Setup (Terminal):
  python3 -m venv ~/rukopys-env && source ~/rukopys-env/bin/activate
  pip install huggingface_hub pillow tqdm
  hf auth login            # optional; needed only if the download says 401/403/gated

Run:
  python make_crops_mac.py                 # gold + test + silver (needs ~25 GB free on the Mac)
  python make_crops_mac.py --no-silver     # gold + test only, much smaller and faster
"""
import argparse, os, tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=str(Path.home() / "rukopys_work"))
ap.add_argument("--no-silver", action="store_true")
ap.add_argument("--keep-raw", action="store_true", help="do not delete downloaded silver page images")
a = ap.parse_args()

WORK = Path(a.workdir); WORK.mkdir(parents=True, exist_ok=True)
DATA, CROPS, CROPS_TAR = WORK / "rukopys", WORK / "crops", WORK / "crops.tar"
USE_SILVER, VAL_FRAC, SEED, MAX_W = not a.no_silver, 0.2, 42, 1536
HAVE_CROPS = False

patterns = ["train/*", "test/*", "*.csv"] + (["silver/*"] if USE_SILVER else [])
snapshot_download("UkrainianCatholicUniversity/rukopys", repo_type="dataset",
                  local_dir=str(DATA), allow_patterns=patterns, max_workers=8)
print("metadata found:", sorted(str(p.relative_to(DATA)) for p in DATA.rglob("metadata.jsonl")))

# ---- identical to cell 5 of the Colab notebook ----
import json, random, re, shutil
from PIL import Image, ImageOps
from tqdm.auto import tqdm

TEXT_TYPES = {'handwritten', 'printed', 'annotation'}
_STRIKE_FIX = re.compile(r'~~(.*?)~~\{(.*?)\}')
_STRIKE = re.compile(r'~~(.*?)~~')
_TABLE = str.maketrans({'—': '-', '–': '-', '«': '"', '»': '"', '“': '"', '”': '"', '„': '"',
                        '‘': "'", '’': "'", 'c': 'с', 'o': 'о', 'p': 'р', 'x': 'х'})
def normalize(t):
    t = _STRIKE_FIX.sub(r'\2', t or '')   # ~~old~~{new} -> new
    t = _STRIKE.sub(r'\1', t)             # ~~text~~ -> text
    return ' '.join(t.translate(_TABLE).split())

def keep(r):
    if r.get('type') not in TEXT_TYPES or r.get('language', 'uk') != 'uk': return False
    if r.get('legibility') == 'illegible': return False
    t = normalize(r.get('text'))
    if not t or t.startswith('[ERROR:'): return False
    x1, y1, x2, y2 = r['bbox']
    return (x2 - x1) >= 4 and (y2 - y1) >= 4

def load_page(path, W, H):
    im = Image.open(path)
    if W and H and im.size != (W, H):          # phone photos: bboxes may refer to EXIF-rotated image
        t = ImageOps.exif_transpose(im)
        if t.size == (W, H): im = t
    return im.convert('RGB')

def crop_split(split):
    meta = DATA / split / 'metadata.jsonl'
    if not meta.exists(): print('no', meta); return []
    out = CROPS / split; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in tqdm(meta.read_text(encoding='utf-8').splitlines(), desc=split):
        page = json.loads(line)
        regs = [r for r in (page.get('regions') or []) if keep(r)]
        if not regs: continue
        fn = page.get('file_name') or page.get('image')
        try:
            im = load_page(DATA / split / fn, page.get('image_width'), page.get('image_height'))
        except Exception as e:
            print('skip', fn, e); continue
        stem = Path(fn).stem
        for i, r in enumerate(regs):
            x1, y1, x2, y2 = map(int, r['bbox'])
            c = im.crop((max(0, x1 - 4), max(0, y1 - 4), min(im.width, x2 + 4), min(im.height, y2 + 4)))
            if c.width > MAX_W:
                c = c.resize((MAX_W, max(1, round(c.height * MAX_W / c.width))))
            p = out / f'{stem}_{i}.jpg'
            c.save(p, quality=92)
            rows.append({'path': str(p.relative_to(CROPS)), 'text': normalize(r['text']),
                         'page': stem, 'type': r['type']})
    return rows

def dump(rows, name):
    with open(CROPS / name, 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{name}: {len(rows)} crops')

if not HAVE_CROPS:
    CROPS.mkdir(exist_ok=True)
    gold = crop_split('train')
    assert gold, 'No gold crops: the dataset was not downloaded correctly (see cell 4 output).'
    pages = sorted({r['page'] for r in gold}); random.Random(SEED).shuffle(pages)
    val_pages = set(pages[:int(len(pages) * VAL_FRAC)])
    dump([r for r in gold if r['page'] not in val_pages], 'gold_train.jsonl')
    dump([r for r in gold if r['page'] in val_pages], 'gold_val.jsonl')
    dump(crop_split('test'), 'test.jsonl')   # final check only (empty if test has no labels)
    if USE_SILVER:
        dump(crop_split('silver'), 'silver.jsonl')
        if not a.keep_raw: shutil.rmtree(DATA / 'silver' / 'images', ignore_errors=True)
    with tarfile.open(CROPS_TAR, 'w') as t:   # backup to Drive (JPEGs: no point compressing)
        t.add(CROPS, arcname='crops')
    print('crop archive on Drive:', round(CROPS_TAR.stat().st_size / 1e9, 2), 'GB')

print("Done. Upload this file to Google Drive -> MyDrive/rukopys_trocr/crops.tar:")
print(CROPS_TAR)
