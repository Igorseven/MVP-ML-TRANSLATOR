"""
Controller responsável pelas rotas de predição
"""
from flask import jsonify, request
from datetime import datetime
from backend.services.model_service import model_service
from backend.config.settings import ERROR_MESSAGES, MAX_BATCH_SIZE, logger
from backend.utils.text_preprocessor import validate_text


def predict():
    """Endpoint para predição de um único texto"""
    try:
        # Verificar se modelo está carregado
        if not model_service.is_loaded():
            return jsonify({
                'error': 'Modelo não carregado',
                'message': ERROR_MESSAGES['MODEL_NOT_LOADED']
            }), 500
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Dados inválidos',
                'message': ERROR_MESSAGES['INVALID_DATA']
            }), 400
        
        if 'Text' not in data:
            return jsonify({
                'error': 'Campo obrigatório ausente',
                'message': ERROR_MESSAGES['MISSING_TEXT']
            }), 400
        
        text = data['Text']
        
        # Validar texto
        is_valid, error_msg = validate_text(text)
        if not is_valid:
            return jsonify({
                'error': 'Texto inválido',
                'message': error_msg
            }), 400
        
        # Fazer predição
        result = model_service.predict_single(text)
        
        if result['error']:
            return jsonify({
                'error': 'Erro na predição',
                'message': result['message']
            }), 500
        
        # Remover campo de erro do resultado
        del result['error']
        
        logger.info(f"Predição realizada: {result['prediction']} (confiança: {result['confidence']}%)")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro no endpoint /predict: {e}")
        return jsonify({
            'error': 'Erro interno do servidor',
            'message': str(e)
        }), 500