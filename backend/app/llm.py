import httpx
from .config import GROQ_API_KEY, GROQ_MODEL

GROQ_URL='https://api.groq.com/openai/v1/chat/completions'

def available(): return bool(GROQ_API_KEY)

async def chat(messages, temperature=0.1, max_tokens=1200):
    if not GROQ_API_KEY: return None
    payload={'model':GROQ_MODEL,'messages':messages,'temperature':temperature,'max_tokens':max_tokens}
    headers={'Authorization':f'Bearer {GROQ_API_KEY}','Content-Type':'application/json'}
    async with httpx.AsyncClient(timeout=45) as client:
        response=await client.post(GROQ_URL,headers=headers,json=payload)
        if response.status_code>=400:
            detail='Groq request failed'
            try: detail=response.json().get('error',{}).get('message',detail)
            except Exception: pass
            raise RuntimeError(f'{detail} (HTTP {response.status_code})')
        data=response.json()
        return data['choices'][0]['message']['content']
