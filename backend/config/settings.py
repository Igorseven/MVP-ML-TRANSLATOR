"""
Configurações do projeto
"""
import os
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações do servidor
HOST = '0.0.0.0'
PORT = 5000

# Caminhos dos arquivos
MODEL_PATH = 'best_language_detection_model.pkl'
MODEL_INFO_PATH = 'model_info.pkl'

# Limites da API
MAX_BATCH_SIZE = 100

# Mensagens de erro padrão
ERROR_MESSAGES = {
    'MODEL_NOT_LOADED': 'O modelo de ML não foi carregado corretamente',
    'INVALID_DATA': 'Requisição deve conter JSON válido',
    'MISSING_TEXT': 'Campo "Text" é obrigatório',
    'INVALID_TEXT': 'Campo "Text" deve ser uma string não vazia',
    'INTERNAL_ERROR': 'Erro interno do servidor',
    'MISSING_TEXTS': 'Campo "texts" é obrigatório',
    'INVALID_FORMAT': 'Campo "texts" deve ser uma lista',
    'TOO_MANY_TEXTS': 'Máximo de 100 textos por requisição',
    'NOT_FOUND': 'Endpoint não encontrado',
    'METHOD_NOT_ALLOWED': 'Método não permitido',
    'MODEL_INFO_NOT_AVAILABLE': 'Informações do modelo não disponíveis'
}

# Configurações de resposta
RESPONSE_LABELS = {
    'Portugese': 'Português',
    'English': 'Inglês'
} 