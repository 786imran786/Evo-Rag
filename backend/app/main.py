import time, socket, ipaddress
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import FRONTEND_ORIGIN, GROQ_MODEL, GROQ_API_KEY
from .schemas import AskRequest, UrlRequest, FeedbackRequest
from .store import documents, chunks, delete_document, clear_all
from .ingest import extract_file, ingest_text
from .pipeline import ask as run_pipeline
from .llm import available

app=FastAPI(title='EvoRAG API',version='1.0.0',description='Adaptive evidence-first RAG API')
app.add_middleware(CORSMiddleware,allow_origins=[FRONTEND_ORIGIN,'http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])

@app.get('/api/health')
def health():
    return {'status':'ok','groq_configured':available(),'model':GROQ_MODEL if available() else None,'documents':len(documents()),'chunks':len(chunks())}

@app.get('/api/documents')
def list_documents():
    return {'documents':documents(),'chunks':len(chunks())}

@app.post('/api/documents/upload')
async def upload_document(file: UploadFile=File(...)):
    data=await file.read()
    if len(data)>12*1024*1024: raise HTTPException(413,'File too large. Maximum size is 12 MB.')
    try: text=extract_file(file.filename or 'upload.txt',data); result=ingest_text(file.filename or 'upload',file.filename or 'upload',text,'file')
    except ValueError as e: raise HTTPException(400,str(e))
    except Exception as e: raise HTTPException(400,f'Could not parse file: {e}')
    return result

def validate_public_url(url):
    p=urlparse(str(url))
    if p.scheme not in ('http','https') or not p.hostname: raise ValueError('Only public HTTP(S) URLs are allowed.')
    host=p.hostname.lower()
    if host in ('localhost','localhost.localdomain') or host.endswith('.local'): raise ValueError('Local URLs are blocked.')
    try:
        infos=socket.getaddrinfo(host,None)
        for info in infos:
            ip=ipaddress.ip_address(info[4][0])
            if not ip.is_global: raise ValueError('Private or local network addresses are blocked.')
    except socket.gaierror: raise ValueError('Could not resolve URL hostname.')

@app.post('/api/documents/url')
async def ingest_url(body: UrlRequest):
    try: validate_public_url(str(body.url))
    except ValueError as e: raise HTTPException(400,str(e))
    try:
        async with httpx.AsyncClient(follow_redirects=False,timeout=12,headers={'User-Agent':'EvoRAG/1.0 research demo'}) as client:
            response=await client.get(str(body.url))
            if response.status_code>=400: raise HTTPException(400,f'URL returned HTTP {response.status_code}.')
            ctype=response.headers.get('content-type','')
            if 'text/html' not in ctype and 'text/plain' not in ctype: raise HTTPException(400,'URL must serve HTML or plain text.')
            if len(response.content)>4*1024*1024: raise HTTPException(413,'Web page is too large (4 MB maximum).')
        soup=BeautifulSoup(response.text,'html.parser')
        for tag in soup(['script','style','nav','footer','header','noscript','svg']): tag.decompose()
        title=soup.title.get_text(' ',strip=True) if soup.title else str(body.url)
        text=soup.get_text(' ',strip=True)
        return ingest_text(title,str(body.url),text,'url')
    except HTTPException: raise
    except Exception as e: raise HTTPException(400,f'Could not ingest URL: {e}')

@app.delete('/api/documents/{doc_id}')
def remove_document(doc_id:str):
    if not delete_document(doc_id): raise HTTPException(404,'Document not found.')
    return {'deleted':True,'document_id':doc_id}

@app.delete('/api/documents')
def remove_all_documents():
    clear_all(); return {'deleted':True}

@app.post('/api/ask')
async def ask_question(body:AskRequest):
    started=time.perf_counter()
    try: result=await run_pipeline(body.question,body.top_k,body.adaptive)
    except RuntimeError as e: raise HTTPException(502,str(e))
    result['metrics']['latency_ms']=round((time.perf_counter()-started)*1000)
    return result

@app.post('/api/feedback')
def feedback(body:FeedbackRequest):
    # Lightweight demo feedback endpoint; wire to durable storage for production.
    from datetime import datetime,timezone
    import json
    from .config import DATA_DIR
    path=DATA_DIR/'feedback.jsonl'
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps({**body.model_dump(),'created_at':datetime.now(timezone.utc).isoformat()})+'\n')
    return {'saved':True}

@app.get('/api/evaluation')
def evaluation():
    docs=documents(); ch=chunks()
    return {'summary':{'documents':len(docs),'chunks':len(ch),'indexed_characters':sum(d['characters'] for d in docs),'groq_configured':available()},'benchmarks':[{'name':'Retrieval availability','value':100 if ch else 0,'unit':'%','description':'Corpus has indexed chunks.'},{'name':'Corpus diversity','value':len(set(c['document_id'] for c in ch)),'unit':'documents','description':'Distinct documents represented in chunks.'},{'name':'Evidence readiness','value':min(100,round(len(ch)/10*100)),'unit':'score','description':'Demo heuristic based on corpus size; not model accuracy.'}],'notice':'These dashboard values are operational indicators, not a validated accuracy or hallucination benchmark. Add a labeled evaluation dataset for scientific comparison.'}
