import pandas as pd
import time
import os
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pysentimiento import create_analyzer

# --- Forçar uso da CPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- 1. CONFIGURAÇÃO E DADOS ---
try:
    df_teste = pd.read_csv('comentarios_curso.csv')
    textos_teste = df_teste['comentario'].tolist()
    rotulos_reais_brutos = df_teste['rotulo'].tolist()
    rotulos_reais = [r.title() if isinstance(r, str) else str(r) for r in rotulos_reais_brutos]
    print(f"Dados de teste carregados: {len(textos_teste)} amostras.")
except FileNotFoundError:
    print("ERRO: O arquivo 'comentarios_curso.csv' não foi encontrado.")
    exit()
except KeyError:
    print("ERRO: O arquivo CSV deve ter as colunas 'comentario' e 'rotulo'.")
    exit()

# --- 2. MODELOS A SEREM TESTADOS ---
modelos_candidatos = {
    "TabularisAI (Multilingue)": "tabularisai/multilingual-sentiment-analysis",
    "Nlptown (5 Estrelas)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "Pysentimiento (PT-BR)": "pysentimiento/bertweet-pt-sentiment"
}

# --- 3. FUNÇÕES DE MAPEAMENTO ---
def mapear_nlptown(label):
    try:
        estrelas = int(str(label).split()[0])
    except ValueError:
        estrelas = 3
    if estrelas <= 2:
        return 'Negativo'
    elif estrelas == 3:
        return 'Neutro'
    else:
        return 'Positivo'

def mapear_tabularisai(label):
    label = str(label).lower()
    if 'positive' in label:
        return 'Positivo'
    elif 'negative' in label:
        return 'Negativo'
    elif 'neutral' in label:
        return 'Neutro'
    else:
        return 'Neutro'

def mapear_pysentimiento(label):
    label = str(label).upper()
    if label == 'POS':
        return 'Positivo'
    elif label == 'NEG':
        return 'Negativo'
    elif label == 'NEU':
        return 'Neutro'
    else:
        return 'Neutro'

# --- 4. EXECUÇÃO DO BENCHMARK ---
resultados_finais = {}

for nome, path in modelos_candidatos.items():
    print(f"\n--- Testando {nome} ---")
    previsoes = []
    tempos = []

    if 'Pysentimiento' in nome:
        try:
            modelo = create_analyzer(task="sentiment", lang="pt")
            print("INFO: Usando PySentimiento.")
        except Exception as e:
            print(f"ERRO ao carregar PySentimiento: {e}")
            continue
    else:
        try:
            modelo = pipeline("text-classification", model=path, max_length=512, truncation=True, device=-1)
        except Exception as e:
            print(f"ERRO ao carregar modelo Hugging Face ({nome}): {e}")
            continue

    for texto in textos_teste:
        start_time = time.time()
        if 'Pysentimiento' in nome:
            resultado = modelo.predict(texto)
            label = resultado.output
        else:
            resultado = modelo(texto)[0]
            label = resultado['label']
        tempos.append(time.time() - start_time)

        if 'Nlptown' in nome:
            previsoes.append(mapear_nlptown(label))
        elif 'TabularisAI' in nome:
            previsoes.append(mapear_tabularisai(label))
        elif 'Pysentimiento' in nome:
            previsoes.append(mapear_pysentimiento(label))
        else:
            previsoes.append(label)

    latencia_media = (sum(tempos) / len(tempos)) * 1000 if tempos else 0.0
    acuracia = accuracy_score(rotulos_reais, previsoes)
    f1_medio = f1_score(rotulos_reais, previsoes, average='weighted', zero_division=0)

    resultados_finais[nome] = {
        "Acurácia": f"{acuracia*100:.1f}%",
        "F1-Score": f"{f1_medio:.3f}",
        "Latência Média (ms)": f"{latencia_media:.2f}"
    }

    print(f"Acurácia: {resultados_finais[nome]['Acurácia']}")
    print(f"F1-Score: {resultados_finais[nome]['F1-Score']}")
    print(f"Latência Média: {resultados_finais[nome]['Latência Média (ms)']} ms")
    print("\nRelatório de Classificação:")
    print(classification_report(rotulos_reais, previsoes, zero_division=0))

# --- 5. RESUMO FINAL E EXPORTAÇÃO ---
print("\n" + "="*50)
print("RESUMO FINAL")
print("="*50)
df_resultados = pd.DataFrame(resultados_finais).T
print(df_resultados.to_markdown(numalign="left", stralign="left"))

# Salvar em CSV
df_resultados.to_csv("resultados_benchmark.csv", index=True)
print("\nResultados salvos em 'resultados_benchmark.csv'")