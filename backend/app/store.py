import json, uuid
from datetime import datetime, timezone
from .config import DOCS_FILE, CHUNKS_FILE

def now(): return datetime.now(timezone.utc).isoformat()
def _read(path, default):
    try:
        with path.open('r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default
def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2)
    temp.replace(path)
def documents(): return _read(DOCS_FILE, [])
def chunks(): return _read(CHUNKS_FILE, [])
def save_document(name, source, text, kind='file'):
    docs=documents(); item={'id':str(uuid.uuid4()),'name':name,'source':source,'kind':kind,'characters':len(text),'created_at':now()}
    docs.insert(0,item); _write(DOCS_FILE,docs); return item
def save_chunks(new_chunks):
    current=chunks(); current.extend(new_chunks); _write(CHUNKS_FILE,current)
def delete_document(doc_id):
    docs=documents(); kept=[d for d in docs if d['id'] != doc_id]
    if len(kept)==len(docs): return False
    _write(DOCS_FILE,kept); _write(CHUNKS_FILE,[c for c in chunks() if c['document_id'] != doc_id]); return True
def clear_all(): _write(DOCS_FILE,[]); _write(CHUNKS_FILE,[])
