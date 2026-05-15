const state = { file: null, algorithm: null, results: null, algorithms: [] };
const ALGO_ICONS = {
  zeroR: "0️⃣",
  oneR: "1️⃣",
  j48: "🌳",
  naiveBayes: "🧪",
  random: "🌲",
  regresionMultiple: "📈",
  rLogistica: "📉",
  series: "🧠",
  kmeans: "🎯",
  em: "🧬"
};

const RING_LENGTH = 263.9;

document.addEventListener('DOMContentLoaded', () => {
  initFileUpload();
  initEvalMethodToggle();
  loadAlgorithms();
});

async function loadAlgorithms() {
  try {
    const resp = await fetch('/api/weka/algorithms');
    if (!resp.ok) throw new Error('Failed to load algorithms');
    const algorithms = await resp.json();
    state.algorithms = algorithms;
    renderAlgorithmGrid(algorithms);
    if (algorithms.length > 0) selectAlgorithm(algorithms[0].id);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

function renderAlgorithmGrid(algorithms) {
  const grid = document.getElementById('algorithm-grid');
  grid.innerHTML = '';
  algorithms.forEach((algorithm) => {
    const card = document.createElement('div');
    card.className = 'algorithm-card';
    card.dataset.algoId = algorithm.id;
    card.innerHTML = `
      <div class="algo-icon">${ALGO_ICONS[algorithm.id] || '⚙️'}</div>
      <div class="algo-name">${algorithm.name}</div>
      <div class="algo-desc">${algorithm.description}</div>
      <div class="algo-tag">${algorithm.mode}</div>
    `;
    card.onclick = () => selectAlgorithm(algorithm.id);
    grid.appendChild(card);
  });
}

function selectAlgorithm(id) {
  state.algorithm = id;
  document.querySelectorAll('.algorithm-card').forEach((card) => {
    card.classList.toggle('selected', card.dataset.algoId === id);
  });
}

function initFileUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  dropZone.onclick = () => fileInput.click();
  fileInput.onchange = (event) => handleFile(event.target.files[0]);

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add('dragging');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove('dragging');
    });
  });

  dropZone.addEventListener('drop', (event) => {
    const [file] = event.dataTransfer.files;
    handleFile(file);
  });
}

function handleFile(file) {
  if (!file) return;
  const lowerName = file.name.toLowerCase();
  if (!lowerName.endsWith('.csv') && !lowerName.endsWith('.arff')) {
    showToast('Only .csv and .arff files are supported', 'warning');
    return;
  }

  state.file = file;
  document.getElementById('drop-zone').style.display = 'none';
  document.getElementById('file-info').style.display = 'flex';
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent = `${(file.size / 1024).toFixed(1)} KB`;
  document.getElementById('stat-dataset').textContent = file.name;
}

function clearFile() {
  state.file = null;
  document.getElementById('drop-zone').style.display = '';
  document.getElementById('file-info').style.display = 'none';
  document.getElementById('results-panel').style.display = 'none';
  document.getElementById('results-empty').style.display = '';
  document.getElementById('stat-dataset').textContent = 'No dataset';
}

function initEvalMethodToggle() {
  document.querySelectorAll('input[name="eval-method"]').forEach((radio) => {
    radio.addEventListener('change', () => {
      const selectedMethod = document.querySelector('input[name="eval-method"]:checked').value;
      const isCrossValidation = selectedMethod === 'crossvalidation';
      document.getElementById('cv-settings').style.display = isCrossValidation ? '' : 'none';
      document.getElementById('ps-settings').style.display = isCrossValidation ? 'none' : '';
    });
  });
}

async function runClassification() {
  if (!state.file || !state.algorithm) {
    showToast('Select a file and an algorithm first', 'warning');
    return;
  }

  const formData = new FormData();
  formData.append('file', state.file);
  formData.append('algorithm', state.algorithm);
  formData.append('evaluationMethod', document.querySelector('input[name="eval-method"]:checked').value);
  formData.append('folds', document.getElementById('folds').value);
  formData.append('trainPercent', document.getElementById('train-percent').value);

  setLoading(true, `Running ${state.algorithm}...`);

  try {
    const response = await fetch('/api/weka/classify', { method: 'POST', body: formData });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || 'Execution error');
    }

    state.results = payload;
    renderResults(payload);
    showToast('Execution completed', 'success');
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    setLoading(false);
  }
}

function renderResults(result) {
  document.getElementById('results-empty').style.display = 'none';
  document.getElementById('results-panel').style.display = '';

  document.getElementById('stat-instances').textContent = result.numInstances;
  document.getElementById('stat-attributes').textContent = result.numAttributes;

  const mainScore = Number(result.accuracy || 0);
  document.getElementById('stat-accuracy').textContent = `${mainScore.toFixed(2)}%`;
  document.getElementById('accuracy-value').textContent = `${mainScore.toFixed(2)}%`;
  document.getElementById('ring-fill').style.strokeDasharray = RING_LENGTH;
  document.getElementById('ring-fill').style.strokeDashoffset = RING_LENGTH * (1 - Math.max(0, Math.min(mainScore, 100)) / 100);

  document.getElementById('result-algorithm-badge').textContent = `${result.algorithm} • ${result.taskType}`;

  renderSummary(result);
  renderDetails(result);
}

function renderSummary(result) {
  const summary = document.getElementById('summary-metrics');
  const rows = [
    ['Method', result.evaluationMethod || '—'],
    ['Kappa', Number(result.kappa || 0).toFixed(4)],
    ['MAE', Number(result.meanAbsoluteError || 0).toFixed(4)]
  ];

  if (result.taskType === 'clustering' && result.clustering?.silhouette !== null && result.clustering?.silhouette !== undefined) {
    rows.push(['Silhouette', Number(result.clustering.silhouette).toFixed(4)]);
  }
  if (result.taskType === 'regression' && result.regression) {
    rows.push(['R²', Number(result.regression.r2 || 0).toFixed(4)]);
    rows.push(['MSE', Number(result.regression.mse || 0).toFixed(4)]);
  }

  summary.innerHTML = rows.map(([label, value]) => `<div class="detail-row"><span>${label}</span><span>${value}</span></div>`).join('');
}

function renderDetails(result) {
  const classificationCard = document.getElementById('classification-card');
  const metricsContainer = document.getElementById('metrics-table-container');

  if (result.taskType === 'classification' && result.confusionMatrix?.length) {
    classificationCard.style.display = '';
    renderCM(result.confusionMatrix, result.classNames);

    let html = '<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>';
    result.classNames.forEach((className, index) => {
      html += `<tr><td>${className}</td><td>${Number(result.precision[index] || 0).toFixed(4)}</td><td>${Number(result.recall[index] || 0).toFixed(4)}</td><td>${Number(result.fMeasure[index] || 0).toFixed(4)}</td></tr>`;
    });
    html += '</tbody></table>';
    metricsContainer.innerHTML = html;
    return;
  }

  classificationCard.style.display = 'none';
  if (result.taskType === 'clustering' && result.clustering) {
    const labels = result.clustering.clusterLabels || [];
    const sizes = result.clustering.clusterSizes || [];
    let html = '<table><thead><tr><th>Cluster</th><th>Size</th></tr></thead><tbody>';
    labels.forEach((label, index) => {
      html += `<tr><td>${label}</td><td>${sizes[index] || 0}</td></tr>`;
    });
    html += '</tbody></table>';
    metricsContainer.innerHTML = html;
    return;
  }

  if (result.taskType === 'regression' && result.regression) {
    metricsContainer.innerHTML = `
      <table>
        <tbody>
          <tr><th>MAE</th><td>${Number(result.regression.mae || 0).toFixed(4)}</td></tr>
          <tr><th>MSE</th><td>${Number(result.regression.mse || 0).toFixed(4)}</td></tr>
          <tr><th>R²</th><td>${Number(result.regression.r2 || 0).toFixed(4)}</td></tr>
        </tbody>
      </table>
    `;
  }
}

function renderCM(matrix, names) {
  const header = names.map((name) => `<th>${name}</th>`).join('');
  let html = `<table><tr><th></th>${header}</tr>`;
  matrix.forEach((row, index) => {
    const cells = row.map((value) => `<td>${value}</td>`).join('');
    html += `<tr><th>${names[index]}</th>${cells}</tr>`;
  });
  html += '</table>';
  document.getElementById('confusion-matrix-container').innerHTML = html;
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 250);
  }, 2800);
}

function setLoading(isVisible, text = 'Processing model...') {
  document.getElementById('loading-overlay').style.display = isVisible ? 'flex' : 'none';
  document.getElementById('loading-text').textContent = text;
}
