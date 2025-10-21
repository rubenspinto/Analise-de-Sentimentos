from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API de Análise de Sentimentos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregando o modelo multilíngue para análise de sentimentos
modelo = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

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
        "confianca": round(resultado["score"], 4)
    }
