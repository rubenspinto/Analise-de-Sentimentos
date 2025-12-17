⚛️| Análise de Sentimentos - Social Brasilis

![Status do Projeto](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Model](https://img.shields.io/badge/AI%20Model-BERTweet%20PT_BR-yellow)

Este projeto é uma ferramenta de Processamento de Linguagem Natural (NLP) desenvolvida para a Social Brasilis. O sistema automatiza a classificação de feedbacks de alunos (Ciclo 2 e posteriores) em POSITIVO, NEGATIVO ou NEUTRO, agilizando a avaliação de impacto dos programas de formação.

---

## 📋| Funcionalidades

```
* Upload de Planilhas: Interface web simples para envio de arquivos `.xlsx, e ods`.
* Classificação Automática: Processamento em lote de comentários utilizando Deep Learning.
* Exportação de Resultados: Gera uma cópia da planilha original com uma nova coluna de sentimentos classificados.
* Alta Precisão: Modelo otimizado para a língua portuguesa (PT-BR) e linguagem informal/redes sociais.
```

---

## 📊| Métricas e Performance

O modelo escolhido para produção foi o **Pysentimiento (BERTweet PT-BR)** após testes comparativos de benchmarking:

```
| Modelo | Acurácia | F1-Score | Latência Média |
| :--- | :--- | :--- | :--- |
| **Pysentimiento (PT-BR)** | **81.2%** | **0.825** | **170.72ms** |
| Nlptown (Multilingue) | 78.1% | 0.803 | 195.47ms |
| TabularisAI | 77.1% | 0.794 | 155.65ms |
```

> *Testes realizados com dataset de validação de 100 frases.*

---

## 💻| Tecnologias Utilizadas

```
* Linguagem: Python 3.10
* Framework Web: FastAPI
* Containerização: Docker & Docker Compose
* IA / NLP: Transformers (Hugging Face), Pytorch
* Modelo Base: `pysentimiento/bertweet-pt-sentiment`
* Processamento de Dados:** Pandas, OpenPyXL
* Frontend: HTML5, TailwindCSS
```

---

## ➡️| Como Executar o Projeto

### Pré-requisitos

```
* [Docker](https://www.docker.com/) e Docker Compose instalados.

 Passo a Passo

1.  Clone o repositório:
    
    git clone [https://github.com/rubenspinto/Analise-de-Sentimentos.git](https://github.com/rubenspinto/Analise-de-Sentimentos.git)
    cd Analise-de-Sentimentos

2.  Suba o ambiente com Docker Compose:
    Este comando irá construir a imagem e iniciar o servidor na porta 7860.
    
    docker-compose up -d --build

3.  Acesse a Aplicação:
    Abra o navegador e vá para:
    `http://localhost:7860/`
```

---

## 📝| Guia de Utilização

```
1.  Formato do Arquivo: O sistema aceita apenas arquivos Excel (.xlsx).
2.  Estrutura Obrigatória: A planilha deve conter os comentários na segunda aba (índice 1).
3.  Coluna Alvo: O sistema busca automaticamente pela coluna com o cabeçalho exato:
    "Qual sua mensagem, dica, sugestão ou crítica para o programa?"
4.  **Resultado: O download iniciará automaticamente com o arquivo `planilha_analisada.xlsx`, contendo a coluna extra "Sentimento dos Alunos".
```
---

## 🗂️| Estrutura do Projeto

```
  ├── app/
  ├── main.py # Lógica principal da API e Modelo IA 
  │ └── frontend.html # Interface do usuário 
  ├── Dockerfile # Configuração da imagem Docker 
  ├── docker-compose.yaml # Orquestração do container 
  ├── requirements.txt # Dependências Python 
  └── README.md # Documentação
```

---

## 👥| Equipe Responsável

Projeto Social Brasilis - Turma E1_2

```
* Rubens Pinto
* João Paulo
* Antonio Franklin
* Ana Cassia
* Gideão Ferreira
* Pedro Henrique
```
---

## 📈| Visualização (Dashboard)

Os dados processados por esta ferramenta alimentam o dashboard de avaliação de impacto:
[Acessar Dashboard no Looker Studio](https://lookerstudio.google.com/reporting/6c022ba

