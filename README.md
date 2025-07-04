# API de Identificação de Idioma

API REST para identificação de idioma (Português ou Inglês) em um texto, utilizando Machine Learning, com um frontend integrado para testes manuais.

## 📋 Estrutura do Projeto

```
ML-TRANSLATOR/
├── backend/                # Código-fonte do backend
│   ├── config/             # Configurações da aplicação
│   ├── controllers/        # Controllers (rotas da API)
│   ├── services/           # Lógica de negócio e serviços
│   ├── tests/              # Testes automatizados
│   └── utils/              # Funções utilitárias
├── frontend/               # Arquivos do frontend (HTML, CSS, JS)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── app.py                  # Ponto de entrada da aplicação
├── best_language_detection_model.pkl # Modelo treinado
├── model_info.pkl          # Informações do modelo
├── dataset_filtered.csv    # Dataset utilizado para o treinamento
├── ML_LANGUAGE_IDENTIFICATION.ipynb # Notebook com código de treinamento
└── requirements.txt        # Dependências
```

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd ML-TRANSLATOR
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
```

3. Ative o ambiente virtual:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🔧 Execução

Execute a aplicação:
```bash
python app.py
```

O comando irá:
1.  Iniciar o servidor backend.
2.  Abrir automaticamente o frontend no seu navegador em `http://localhost:5000`.

## 📌 Endpoints da API

Todas as rotas da API estão disponíveis sob o prefixo `/api`. O frontend já está configurado para usá-las.

### Health Check
- `GET /api` - Status da API
- `GET /api/health` - Health check

### Predição
- `POST /api/predict` - Classificar um texto

### Exemplo de Requisição

#### Predição única:
```json
POST /api/predict
{
    "Text": "Este é um texto de exemplo."
}
```

#### Resposta:
```json
{
    "text": "Este é um texto de exemplo.",
    "prediction": "Português",
    "language": "Português",
    "confidence": 98.7,
    "processed_text": "este é um texto de exemplo",
    "timestamp": "2024-01-01T12:00:00"
}
```

## 🧪 Testes

O projeto possui uma suíte de testes robusta utilizando **PyTest** para garantir a qualidade e o desempenho da aplicação.

### Tipos de Testes

1.  **Testes de Desempenho do Modelo** (`tests/test_model_performance.py`):
    -   **O quê?** Avalia o modelo de Machine Learning (`.pkl`) contra um conjunto de dados de teste.
    -   **Para quê?** Garante que o modelo atenda a métricas mínimas de qualidade (acurácia, precisão, recall, F1-score) antes de ir para produção.

2.  **Testes de Integração** (`tests/test_controllers.py`):
    -   **O quê?** Testa os endpoints da API (`/api/*`).
    -   **Para quê?** Simula requisições HTTP e valida as respostas, garantindo que as rotas funcionem como esperado.

3.  **Testes Unitários** (`tests/test_model_service.py`):
    -   **O quê?** Testa as funções e classes de forma isolada, como a `ModelService`.
    -   **Para quê?** Verifica a lógica interna de cada componente, como a predição e o tratamento de dados.

### Como Executar os Testes

Para facilitar a execução, utilize o script `run_tests.py`:

```bash
# Executar todos os testes
python run_tests.py

# Executar apenas os testes de desempenho do modelo
python run_tests.py performance

# Executar testes com relatório de cobertura de código
python run_tests.py coverage
```

## 🏗️ Arquitetura

O projeto segue uma arquitetura em camadas, modularizada dentro da pasta `backend`:

1.  **Frontend**: Interface do usuário (`frontend/`) para interação com a API.
2.  **Backend**: Aplicação principal (`backend/`) que contém:
    -   **Controllers**: Responsáveis por receber requisições HTTP da API e retornar respostas.
    -   **Services**: Contém a lógica de negócio (carregamento e uso do modelo ML).
    -   **Utils**: Funções auxiliares reutilizáveis.
    -   **Config**: Configurações centralizadas da aplicação.

Essa estrutura facilita a manutenção, testabilidade e escalabilidade do código.

## 🤖 Sobre o Modelo

O modelo de Machine Learning foi treinado no **Google Colab** e todo o código de treinamento, análise exploratória e avaliação está disponível no arquivo `ML_LANGUAGE_IDENTIFICATION.ipynb`. Este notebook contém:

- Análise exploratória dos dados
- Pré-processamento e limpeza de texto
- Treinamento e comparação de diferentes algoritmos
- Avaliação de métricas de desempenho
- Exportação do modelo final

## 📝 Notas

- Os modelos (`best_language_detection_model.pkl` e `model_info.pkl`) devem estar presentes no diretório raiz.