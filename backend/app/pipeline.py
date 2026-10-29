import json, re
from .retrieval import retrieve
from .llm import chat, available

SYSTEM="You are EvoRAG, an evidence-first research assistant. Use ONLY the supplied evidence for factual claims about the user's corpus. Cite supporting chunks inline as [S1], [S2]. If evidence is insufficient or contradictory, say so plainly. Separate directly supported facts from cautious inference. Never invent sources or citations. Keep the answer useful and concise."

async def ask(question, top_k=5, adaptive=True):
    plan={'query_type':'factual','subqueries':[],'knowledge_gap':False,'strategy':'semantic + metadata-aware retrieval'}
    if available():
        try:
            raw=await chat([{'role':'system','content':'Analyze the question for retrieval. Return only JSON with keys query_type (factual, comparison, explanation, temporal, multi-hop), subqueries (up to 3 short search queries), and knowledge_gap_hypothesis (one sentence). Do not answer the question.'},{'role':'user','content':question}],temperature=0,max_tokens=250)
            match=re.search(r'\{.*\}',raw,re.S); parsed=json.loads(match.group()) if match else {}
            plan.update({k:parsed[k] for k in ('query_type','subqueries','knowledge_gap_hypothesis') if k in parsed})
        except Exception: pass
    queries=[question]+[q for q in plan.get('subqueries',[]) if isinstance(q,str) and q.strip()][:2]
    all_hits={}
    meta=[]
    for q in queries:
        hits,m=retrieve(q,top_k)
        meta.append(m)
        for hit in hits:
            prev=all_hits.get(hit['id'])
            if prev is None or hit['score']>prev['score']: all_hits[hit['id']]=hit
    evidence=sorted(all_hits.values(),key=lambda x:x['score'],reverse=True)[:top_k]
    best=evidence[0]['score'] if evidence else 0
    gap=(not evidence) or best<0.12
    plan['knowledge_gap']=gap
    plan['retrieval_queries']=queries
    plan['adaptive_action']='Broadened retrieval with query decomposition' if len(queries)>1 else ('Flagged weak evidence' if gap else 'Standard evidence retrieval')
    if not evidence:
        return {'answer':'I do not have indexed evidence for this question yet. Add relevant documents or a public URL to the Knowledge Base, then ask again.','citations':[],'evidence':[],'plan':plan,'verification':{'status':'insufficient_evidence','supported_claims':0,'unsupported_claims':0,'notes':'No indexed evidence was retrieved.'},'metrics':{'retrieval_confidence':0,'evidence_coverage':0,'latency_ms':0,'retrieval_method':'none'},'model':None}
    context='\n\n'.join(f"[S{i+1}] Source: {h['document_name']} | Chunk {h['chunk_index']} | Relevance {h['score']:.3f}\n{h['text']}" for i,h in enumerate(evidence))
    answer=None; model=None
    if available():
        answer=await chat([{'role':'system','content':SYSTEM},{'role':'user','content':f'Question: {question}\n\nEvidence:\n{context}\n\nWrite a grounded answer with citations like [S1]. If the evidence does not answer the question, say what is missing.'}],temperature=0.1,max_tokens=1300)
        model='Groq / configured model'
    else:
        answer='**Groq is not configured yet.** Add `GROQ_API_KEY` to `backend/.env` and restart the API to enable generated answers.\n\nRetrieved evidence:\n\n'+'\n\n'.join(f"**[S{i+1}] {h['document_name']}**\n{h['text'][:700]}" for i,h in enumerate(evidence))
    verification={'status':'needs_review' if gap else 'evidence_attached','supported_claims':None,'unsupported_claims':None,'notes':'Retrieved evidence is attached. Verification is heuristic; inspect the cited passages before relying on high-stakes conclusions.'}
    if available() and answer:
        try:
            verdict=await chat([{'role':'system','content':'Audit the answer against the supplied evidence. Return only JSON: {"status":"supported|partially_supported|unsupported|contradictory", "supported_claims": integer, "unsupported_claims": integer, "notes": "brief explanation"}. Be conservative.'},{'role':'user','content':f'ANSWER:\n{answer}\n\nEVIDENCE:\n{context}'}],temperature=0,max_tokens=220)
            m=re.search(r'\{.*\}',verdict,re.S)
            if m: verification.update(json.loads(m.group()))
        except Exception: pass
    citations=[]
    for i,h in enumerate(evidence): citations.append({'id':f'S{i+1}','document_id':h['document_id'],'document_name':h['document_name'],'source':h['source'],'chunk_index':h['chunk_index'],'score':h['score']})
    coverage=sum(1 for c in citations if f"[{c['id']}]" in answer)/max(1,len(citations))
    return {'answer':answer,'citations':citations,'evidence':[{'id':f'S{i+1}',**h} for i,h in enumerate(evidence)],'plan':plan,'verification':verification,'metrics':{'retrieval_confidence':best,'evidence_coverage':round(coverage,2),'latency_ms':None,'retrieval_method':meta[0]['method'] if meta else 'none'},'model':model}
