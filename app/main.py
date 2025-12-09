from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from starlette.background import BackgroundTask
import os
import pandas as pd



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



def delete_file(path): # Função para deletar arquivo após download

    if os.path.exists(path):
        os.remove(path)

def avaliandoArquivo(df, textos):

    # Separando os textos
    x_campo = textos
    x_campo_lista = x_campo.astype(str).tolist()


    # Fazendo a previsão do modelo
    y_pred_campo = list(modelo(x_campo_lista, batch_size=30))
    predic_campo = [r["label"] for r in y_pred_campo]

    # Adicionando as previsões ao DataFrame
    df["Sentimento dos Alunos"] = predic_campo


    return df

class TextoEntrada(BaseModel):
    texto: str


@app.get("/")
def frontend():
    return FileResponse("app/frontend.html")
    
@app.post("/analisar_arquivo")
async def upload_planilha(file: UploadFile):

    # Verifica o formato do arquivo
    if not file.filename.endswith(".xlsx") and not file.filename.endswith(".ods"):
        return HTMLResponse("<h3>Formato de arquivo não suportado. Por favor, envie um arquivo .xlsx ou .ods</h3>")
    
    # Cria pasta temporária para salvar o arquivo
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file.filename)
    
    try:
        # Salva o file temporariamente no disco
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Lê a planilha
        dfPlanilha = pd.read_excel(temp_path, skiprows=1, sheet_name=1)

        # Preenche valores nulos na coluna de análise de texto
        textosAnalise = dfPlanilha["Qual sua mensagem, dica, sugestão ou crítica para o programa?"].fillna("Normal")

        # Faz a análise de sentimentos
        previsoes = avaliandoArquivo(dfPlanilha, textosAnalise)

        nome, extensao = os.path.splitext(file.filename) # Pega a extensão do arquivo
        output_path = nome + "_com_sentimentos" + extensao   # Nome do arquivo de saída
        previsoes.to_excel(output_path, index=False) # Salva o arquivo de saída

        task = BackgroundTask(delete_file, output_path) # Deleta o arquivo após o download

        if file.filename.endswith(".xlsx"):
            
            # Retorna o arquivo para download em XLSX
            response = FileResponse(
                output_path, 
                filename=output_path, 
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                background=task
            )
            response.set_cookie(key="downloadFinalizado", value="1")

        if file.filename.endswith(".ods"):

            # Retorna o arquivo para download em ODS
            response = FileResponse(
                output_path, 
                filename=output_path, 
                media_type='application/vnd.oasis.opendocument.spreadsheet',
                background=task
            )
            response.set_cookie(key="downloadFinalizado", value="1")

        return response

    except Exception as e:
        # Em caso de erro, retorna uma mensagem de erro
        return HTMLResponse(f"<h3>Erro ao processar o arquivo: {str(e)}</h3>")
    
    finally:
        # Remove o arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
            os.rmdir("temp")
