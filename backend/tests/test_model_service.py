"""
Testes para o ModelService usando PyTest
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import joblib
import numpy as np
from backend.services.model_service import ModelService
from backend.config.settings import RESPONSE_LABELS


class TestModelService:
    """Testes unitários para o ModelService"""
    
    @pytest.fixture
    def service(self):
        """Fixture para criar instância do ModelService"""
        return ModelService()
    
    @patch('backend.services.model_service.joblib.load')
    @patch('backend.services.model_service.os.path.exists')
    def test_load_model_success(self, mock_exists, mock_joblib, service):
        """Testa carregamento bem-sucedido do modelo"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_model = Mock()
        mock_info = {"accuracy": 0.98}
        mock_joblib.side_effect = [mock_model, mock_info]
        
        # Executar
        service.load_model()
        
        # Verificar
        assert service.model is not None
        assert service.model_info == {"accuracy": 0.98}
        assert mock_joblib.call_count == 2
    
    @patch('backend.services.model_service.os.path.exists')
    def test_load_model_file_not_found(self, mock_exists, service):
        """Testa erro quando arquivo do modelo não existe"""
        # Configurar mock
        mock_exists.return_value = False
        
        # Executar e verificar
        with pytest.raises(FileNotFoundError):
            service.load_model()
    
    def test_is_loaded(self, service):
        """Testa verificação se modelo está carregado"""
        # Modelo não carregado
        assert service.is_loaded() is False
        
        # Modelo carregado
        service.model = Mock()
        assert service.is_loaded() is True
    
    def test_get_model_info(self, service):
        """Testa obtenção de informações do modelo"""
        # Sem informações
        assert service.get_model_info() == {}
        
        # Com informações
        service.model_info = {"accuracy": 0.98}
        assert service.get_model_info() == {"accuracy": 0.98}
    
    @patch('backend.services.model_service.preprocess_text')
    def test_predict_single_success_english(self, mock_preprocess, service):
        """Testa predição bem-sucedida de um texto em inglês"""
        # Configurar mocks
        mock_preprocess.return_value = "processed text"
        mock_model = Mock()
        mock_model.predict.return_value = ['English']
        mock_model.predict_proba.return_value = [[0.1, 0.9]]
        service.model = mock_model
        
        # Executar
        result = service.predict_single("Test text")
        
        # Verificar
        assert result['error'] is False
        assert result['prediction'] == RESPONSE_LABELS['English']
        assert result['language'] == 'Inglês'
        assert result['confidence'] == 90.0
        assert result['confidence_method'] == 'probability'
    
    @patch('backend.services.model_service.preprocess_text')
    def test_predict_single_empty_text(self, mock_preprocess, service):
        """Testa predição com texto vazio"""
        # Configurar mock
        mock_preprocess.return_value = ""
        service.model = Mock()
        
        # Executar
        result = service.predict_single("")
        
        # Verificar
        assert result['error'] is True
        assert result['message'] == 'Texto inválido'
    
    def test_calculate_confidence_probability(self, service):
        """Testa cálculo de confiança usando probabilidade"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        service.model = mock_model
        
        # Executar
        result = service._calculate_confidence("texto")
        
        # Verificar
        assert result['confidence'] == 70.0
        assert result['method'] == 'probability'
    
    def test_calculate_confidence_decision_function(self, service):
        """Testa cálculo de confiança usando decision function"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.side_effect = AttributeError()
        mock_model.decision_function.return_value = [2.0]
        service.model = mock_model
        
        # Executar
        result = service._calculate_confidence("texto")
        
        # Verificar
        assert result['confidence'] > 50.0
        assert result['method'] == 'decision_function'
    
    def test_calculate_confidence_default(self, service):
        """Testa cálculo de confiança com valor padrão"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict_proba.side_effect = AttributeError()
        mock_model.decision_function.side_effect = Exception()
        service.model = mock_model
        
        # Executar
        result = service._calculate_confidence("texto")
        
        # Verificar
        assert result['confidence'] == 50.0
        assert result['method'] == 'default'
    
    @pytest.mark.parametrize("prediction,expected_label", [
        ('English', RESPONSE_LABELS['English']),
        ('Portugese', RESPONSE_LABELS['Portugese'])
    ])
    def test_predict_with_different_languages(self, service, prediction, expected_label):
        """Testa predições para diferentes idiomas"""
        # Configurar mock
        mock_model = Mock()
        mock_model.predict.return_value = [prediction]
        mock_model.predict_proba.return_value = [[0.5, 0.5]]
        service.model = mock_model
        
        # Executar
        result = service.predict_single("Teste")
        
        # Verificar
        assert result['prediction'] == expected_label
        assert result['language'] == expected_label 