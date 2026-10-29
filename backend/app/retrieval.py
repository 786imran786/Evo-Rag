import re, math
from collections import Counter
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .store import chunks

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def tokens(s): return re.findall(r'[a-zA-Z0-9_]{2,}',s.lower())
def retrieve(query: str, top_k: int=5):
    corpus=chunks()
    if not corpus: return [], {'method':'none','corpus_chunks':0,'confidence':0.0}
    texts=[c['text'] for c in corpus]
    method='sentence-transformers'
    try:
        model=_model()
        vectors=model.encode([query]+texts,normalize_embeddings=True,show_progress_bar=False)
        scores=np.dot(vectors[1:],vectors[0]).tolist()
    except Exception:
        method='tf-idf-fallback'
        matrix=TfidfVectorizer(ngram_range=(1,2),stop_words='english').fit_transform([query]+texts)
        scores=cosine_similarity(matrix[1:],matrix[0]).ravel().tolist()
    qterms=set(tokens(query))
    ranked=[]
    for chunk,score in zip(corpus,scores):
        overlap=len(qterms & set(tokens(chunk['text']))) / max(1,len(qterms))
        recency=0.0
        # Small metadata-aware boost for source chunks with query terms in title.
        title_bonus=0.025 if qterms & set(tokens(chunk.get('document_name',''))) else 0.0
        final=float(score)+0.06*overlap+title_bonus+recency
        ranked.append((final,float(score),chunk))
    ranked.sort(key=lambda x:x[0],reverse=True)
    chosen=[]; seen=set()
    for final,semantic,chunk in ranked:
        # Diversify across documents when there are alternatives.
        if chunk['document_id'] in seen and len({r['document_id'] for _,_,r in ranked[:top_k]})>1: continue
        chosen.append({**chunk,'score':round(max(0,min(1,final)),4),'semantic_score':round(max(0,min(1,semantic)),4)})
        seen.add(chunk['document_id'])
        if len(chosen)>=top_k: break
    best=chosen[0]['score'] if chosen else 0.0
    return chosen, {'method':method,'corpus_chunks':len(corpus),'confidence':round(best,4)}
