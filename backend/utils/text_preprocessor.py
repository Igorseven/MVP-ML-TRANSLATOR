"""
Utilitários para preprocessamento de texto
"""
import re
import string
import pandas as pd


def preprocess_text(text):
    """
    Preprocessa o texto, convertendo para minúsculas e removendo espaços extras.
    
    Args:
        text: Texto a ser processado
        
    Returns:
        str: Texto processado
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Converter para string e minúsculas
    text = str(text).lower()
    
    # Remover espaços extras
    text = ' '.join(text.split())
    
    return text


def validate_text(text):
    """
    Valida se o texto é válido para processamento.
    
    Args:
        text: Texto a ser validado
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "O texto deve ser uma string não vazia"
    
    processed = preprocess_text(text)
    if not processed.strip():
        return False, "Texto inválido após processamento"
    
    return True, None 