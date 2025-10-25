from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="API de Análise de Sentimentos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregando o modelo multilíngue para análise de sentimentos
modelo = pipeline(
    "text-classification", model="tabularisai/multilingual-sentiment-analysis"
)

# Carregando o modelo Dataset Multilíngue (estrelas 1–5)
# modelo = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


# Carregando o Modelo NILC (USP)
# modelo = pipeline("sentiment-analysis", model="pysentimiento/bertweet-pt-sentiment")


# Carregando o modelo XLM-R (Twitter, multilíngue)
# modelo = pipeline("sentiment-analysis", model="pysentimiento/bertweet-pt-sentiment")


class TextoEntrada(BaseModel):
    texto: str


@app.get("/")
def frontend():
    return FileResponse("app/frontend.html")


@app.post("/analisar")
async def analisar_sentimento(dados: TextoEntrada):
    resultado = modelo(dados.texto)[0]
    return {
        "texto": dados.texto,
        "sentimento": resultado["label"],
        "confianca": round(resultado["score"], 4),
    }
