const API_BASE_URL = 'http://localhost:5000';

function switchTab(tabName) {
   document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
   document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

   event.target.classList.add('active');
   document.getElementById(tabName + 'Tab').classList.add('active');

   resetAnalysis();
}

async function analyzeText() {
   const text = document.getElementById('textInput').value.trim();

   if (!text) {
      showError('singleError', 'Por favor, digite um texto para análise.');
      return;
   }

   hideError('singleError');
   showLoading('singleLoading');
   hideResults('singleResults');

   try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
         method: 'POST',
         headers: {
            'Content-Type': 'application/json',
         },
         body: JSON.stringify({ Text: text })
      });

      if (!response.ok) {
         const errorData = await response.json();
         throw new Error(`Erro na API: ${response.status} - ${errorData.message || response.statusText}`);
      }

      const result = await response.json();

      hideLoading('singleLoading');
      showResult(result);

   } catch (error) {
      hideLoading('singleLoading');
      console.error('Erro:', error);

      if (error.name === 'TypeError' && error.message.includes('fetch')) {
         showError('singleError', 'Não foi possível conectar com o servidor. Verifique se a API está rodando.');
      } else {
         showError('singleError', `Erro ao processar o texto: ${error.message}`);
      }
   }
}

function showResult(result) {
   const language = result.language;
   const icon = document.getElementById('singleResultIcon');
   const textEl = document.getElementById('singleResultText');
   const confidence = document.getElementById('singleConfidence');
   const originalText = document.getElementById('singleText');

   if (language === 'Português') {
      icon.innerHTML = '🇵🇹';
      icon.className = 'result-icon';
      textEl.textContent = 'Idioma: Português';
      textEl.className = 'result-text portuguese';
   } else if (language === 'Inglês') {
      icon.innerHTML = '🇬🇧';
      icon.className = 'result-icon';
      textEl.textContent = 'Idioma: Inglês';
      textEl.className = 'result-text english';
   } else {
      icon.innerHTML = '<i class="fas fa-question-circle"></i>';
      icon.className = 'result-icon';
      textEl.textContent = 'Idioma Desconhecido';
      textEl.className = 'result-text';
   }

   confidence.innerHTML = `<i class="fas fa-chart-line"></i> Confiança: ${result.confidence}%`;
   originalText.textContent = `"${result.text}"`;

   showResults('singleResults');
}

function showLoading(loadingId) {
   document.getElementById(loadingId).style.display = 'block';
}

function hideLoading(loadingId) {
   document.getElementById(loadingId).style.display = 'none';
}

function showResults(resultsId) {
   document.getElementById(resultsId).style.display = 'block';
}

function hideResults(resultsId) {
   document.getElementById(resultsId).style.display = 'none';
}

function showError(errorId, message) {
   const errorElement = document.getElementById(errorId);
   errorElement.textContent = message;
   errorElement.style.display = 'block';
}

function hideError(errorId) {
   document.getElementById(errorId).style.display = 'none';
}

function resetAnalysis() {
   document.getElementById('textInput').value = '';
   hideResults('singleResults');
   hideError('singleError');
   hideLoading('singleLoading');
}

document.addEventListener('DOMContentLoaded', () => {
   const container = document.querySelector('.container');
   container.style.opacity = '0';
   container.style.transform = 'translateY(20px)';

   setTimeout(() => {
      container.style.transition = 'all 0.6s ease';
      container.style.opacity = '1';
      container.style.transform = 'translateY(0)';
   }, 100);
});