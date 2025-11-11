from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from transformers import pipeline
import kagglehub
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score


# Baixando Dataset 
path = kagglehub.dataset_download("augustop/portuguese-tweets-for-sentiment-analysis")

app = FastAPI(title="API de Análise de Sentimentos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregando o Modelo
modelo = pipeline("sentiment-analysis", model="pysentimiento/bertweet-pt-sentiment")

# Funcões e Rotas
# Função para limpar os textos
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove menções
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"[^a-zA-Záéíóúçãõ ]", " ", text)  # Remove caracteres especiais
    text = text.lower()  # Converte para minúsculas
    return text

# Função de avalição dos datasets
def avaliacaoDF():
    # Ler o Arquivo dataset Tweet
    # arquivo = pd.read_csv(os.path.join(path, "TweetsWithTheme.csv"))
    # df = pd.DataFrame(arquivo)
    # abrev = {"Positivo": "POS", "Negativo": "NEG"}
    # df['sentimento'] = df["sentiment"].replace(abrev)

    # Ler o Arquivo dataset Curso
    df = pd.read_csv("app/dataset_comentarios_curso.csv")

    # Ler o Arquivo dataset Prdouto
    # df = pd.read_csv("app/dataset_comentarios_produto.csv")

    #Limpa os textos dataset do Curso e Produto
    df["texto_limpo"] = df["comentario"].apply(clean_text)

    #Limpa os textos do dataset do Tweet
    # df["texto_limpo"] = df["tweet_text"].apply(clean_text)

    #Separando os textos e os labels
    x_teste = df["texto_limpo"].head(1000)
    x_teste_lista = x_teste.dropna().astype(str).tolist()
    y_teste = df["sentimento"].head(1000)


    # Fazendo a previsão do modelo
    y_pred = list(modelo(x_teste_lista, batch_size=30))
    predic = [r["label"] for r in y_pred]

    # Criando DataFrame comparativo
    comparacao = pd.DataFrame({
        "texto": x_teste,
        "sentimento_real": y_teste,
        "sentimento_previsto": predic
    })

    # Filtrando erros
    erros = comparacao[comparacao["sentimento_real"] != comparacao["sentimento_previsto"]]

    #Avaliando a acurácia
    acc = accuracy_score(y_teste, predic)
    return  acc, erros


class TextoEntrada(BaseModel):
    texto: str


@app.get("/")
def frontend():
    return FileResponse("app/frontend.html")

@app.get("/dados")
async def dados():
    try:
        acuracia, errosT = avaliacaoDF()
        return JSONResponse(content={
            "avaliacao": acuracia,
            "erros": jsonable_encoder(errosT.head(10).to_dict(orient="records"))
        })
    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)