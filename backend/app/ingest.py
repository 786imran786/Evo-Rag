import re, uuid
from pathlib import Path
from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .store import save_document, save_chunks

def clean_text(text: str) -> str:
    text=text.replace('\x00',' ')
    text=re.sub(r'[ \t]+',' ',text)
    text=re.sub(r'\n{3,}','\n\n',text)
    return text.strip()

def split_chunks(text: str) -> List[str]:
    text=clean_text(text)
    if not text: return []
    chunks=[]; start=0
    while start < len(text):
        end=min(start+CHUNK_SIZE,len(text))
        if end < len(text):
            boundary=max(text.rfind('\n',start,end),text.rfind('. ',start,end),text.rfind(' ',start,end))
            if boundary > start+int(CHUNK_SIZE*.55): end=boundary+1
        piece=text[start:end].strip()
        if piece: chunks.append(piece)
        if end>=len(text): break
        start=max(start+1,end-CHUNK_OVERLAP)
    return chunks

def ingest_text(name, source, text, kind='file'):
    text=clean_text(text)
    if len(text)<20: raise ValueError('The extracted text is too short to index.')
    doc=save_document(name,source,text,kind)
    rows=[]
    for i,chunk in enumerate(split_chunks(text)):
        rows.append({'id':str(uuid.uuid4()),'document_id':doc['id'],'document_name':name,'source':source,'chunk_index':i,'text':chunk,'created_at':doc['created_at'],'kind':kind})
    save_chunks(rows)
    return {**doc,'chunks_added':len(rows)}

def extract_file(filename: str, content: bytes) -> str:
    ext=Path(filename).suffix.lower()
    if ext in {'.txt','.md','.csv','.json','.py','.html','.log'}:
        return content.decode('utf-8',errors='replace')
    if ext=='.pdf':
        from pypdf import PdfReader
        import io
        reader=PdfReader(io.BytesIO(content))
        return '\n\n'.join(p.extract_text() or '' for p in reader.pages)
    if ext=='.docx':
        from docx import Document
        import io
        doc=Document(io.BytesIO(content))
        return '\n'.join(p.text for p in doc.paragraphs)
    raise ValueError('Unsupported file type. Upload TXT, MD, CSV, JSON, PY, HTML, PDF, or DOCX.')
