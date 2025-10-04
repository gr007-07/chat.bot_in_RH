!pip show google-generativeai

import google.generativeai as genai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

GOOGLE_API_KEY = 'AIzaSyBcfC3aAtJLlHyuU8UleYvB_hWtXtRKK4U'
genai.configure(api_key=GOOGLE_API_KEY)

genai.configure(api_key=GOOGLE_API_KEY)

import pandas as pd

# Base do conhecimento
documentos_rh = [
    {
        'titulo': 'Política de férias',
        'conteudo': 'Todo funcionário com mais de 12 meses de contrato tem direito a 30 dias de férias remuneradas. A solicitação deve ser feita com 60 dias de antecedência. É permitido vender até 10 dias de férias.'
    },
    {
        'titulo': 'Política de Home Office',
        'conteudo': 'O modelo de trabalho padrão é híbrido. Para os dias de home office, a empresa oferece uma ajuda de custo de R$ 100,00 mensais para despesas com internet e energia.'
    },
    {
        'titulo': 'Política de Licença Paternidade',
        'conteudo': 'A empresa oferece uma licença paternidade estendida de 30 dias corridos para novos pais, superior aos 5 dias previstos em lei. A licença deve começar na data de nascimento do bebê.'
    }
]

# Criando o DataFrame
df_docs = pd.DataFrame(documentos_rh)


df_docs

model_chat = genai.GenerativeModel('gemini-2.5-flash')
pergunta_usuario = "Quantos dias de licença paternidade a empresa oferece?"
response_sem_rag = model_chat.generate_content(pergunta_usuario)

print(f"Pergunta: {pergunta_usuario}")
print(f"Resposta do LLM (sem contexto): {response_sem_rag.text}")

import hashlib

def embed_with_cache(content, cache={}):
    key = hashlib.md5(content.encode()).hexdigest()
    if key in cache:
        return cache[key]
    embedding = genai.embed_content(model='models/embedding-001', content=content)['embedding']
    cache[key] = embedding
    return embedding


embedding_pergunta = genai.embed_content(
    model='models/embedding-001',
    content=pergunta_usuario
)['embedding']

df_docs['Embedding'] = df_docs['conteudo'].apply(
    lambda x: genai.embed_content(
        model='models/embedding-001',
        content=x
    )['embedding']
)

df_docs['Similaridade'] = df_docs['Embedding'].apply(
    lambda x: cosine_similarity([x], [embedding_pergunta])[0][0]
)
documento_relevante = df_docs.sort_values(
    by='Similaridade',
    ascending=False
).iloc[0]

prompt_com_rag = f"""
Com base APENAS no contexto abaixo, responda a pergunta do usuário. Se não estiver no contexto, diga "Não encontrei essa informação nos documentos".

**Contexto:**
"{documento_relevante['conteudo']}"

**Pergunta:**
"{pergunta_usuario}
"""


response_com_rag = model_chat.generate_content(prompt_com_rag)
print(response_com_rag.text)

#função de reutilizar código

def chatbot_rh(pergunta):
  embedding_pergunta = genai.embed_content(model='models/embedding-001', content=pergunta_usuario)['embedding']
  df_docs['Similaridade'] = df_docs['Embedding'].apply(lambda x: cosine_similarity([x], [embedding_pergunta])[0][0])
  documento_relevante = df_docs.sort_values(by='Similaridade',ascending=False).iloc[0]
  prompt_final = f"""
  Com base APENAS no contexto abaixo, responda a pergunta do usuário. Se não estiver no contexto, diga "Não encontrei essa informação nos documentos".

**Contexto:**
"{documento_relevante['conteudo']}"

**Pergunta:**
"{pergunta}
"""
  resposta = model_chat.generate_content(prompt_final)
  return resposta.text

nova_pergunta = "A empresa oferece ajuda de custo para home office"

print(f"Pergunta: {nova_pergunta}")
print(f"Resposta: {chatbot_rh(nova_pergunta)}")
