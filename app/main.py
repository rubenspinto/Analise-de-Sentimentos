from fastapi import FastAPI, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from transformers import pipeline
from starlette.background import BackgroundTask
import kagglehub
import os
import pandas as pd
import re



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
def delete_file(path):
    if os.path.exists(path):
        os.remove(path)

def avaliandoArquivo(df, textos):
    # Separando os textos
    x_campo = textos
    x_campo_lista = x_campo.astype(str).tolist()


    # Fazendo a previsão do modelo
    y_pred_campo = list(modelo(x_campo_lista, batch_size=30))
    predic_campo = [r["label"] for r in y_pred_campo]


    df["Sentimento dos Alunos"] = predic_campo


    return df

class TextoEntrada(BaseModel):
    texto: str


@app.get("/")
def frontend():
    return FileResponse("app/frontend.html")
    
@app.post("/analisar_arquivo")
async def upload_planilha(file: UploadFile):
    if not file.filename.endswith(".xlsx"):
        return JSONResponse(content={"erro": "Formato inválido. Use xlsx."}, status_code=400)

    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file.filename)
    
    try:
        # Salva o file temporariamente no disco
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".xlsx"):
            dfPlanilha = pd.read_excel(temp_path, skiprows=1, sheet_name=1)
            textosAnalise = dfPlanilha["Qual sua mensagem, dica, sugestão ou crítica para o programa?"].fillna("Normal")

            previsoes = avaliandoArquivo(dfPlanilha, textosAnalise)
            output_path = "planilha_analisada.xlsx"
            previsoes.to_excel(output_path, index=False)

            task = BackgroundTask(delete_file, output_path)


            response = FileResponse(
                output_path, 
                filename="planilha_analisada.xlsx", 
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                background=task
            )
            response.set_cookie(key="downloadFinalizado", value="1")
        return response

    except Exception as e:
        return JSONResponse(content={"erro": f"Erro ao salvar o arquivo: {str(e)}"}, status_code=500)
    
    finally:
        # Remove o arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
            os.rmdir("temp")
