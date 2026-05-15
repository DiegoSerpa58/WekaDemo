const state = { file: null, algorithm: null, results: null };
const ALGO_ICONS = { "zeroR": "0️⃣", "oneR": "1️⃣", "j48": "🌳", "naiveBayes": "🧪" };

document.addEventListener('DOMContentLoaded', () => {
    initFileUpload();
    initEvalMethodToggle();
    loadAlgorithms();
});

async function loadAlgorithms() {
    try {
        const resp = await fetch('/api/weka/algorithms');
        const algorithms = await resp.json();
        renderAlgorithmGrid(algorithms);
        if (algorithms.length > 0) selectAlgorithm(algorithms[0].id);
    } catch (e) {
        console.error(e);
    }
}

function renderAlgorithmGrid(algorithms) {
    const grid = document.getElementById('algorithm-grid');
    grid.innerHTML = '';
    algorithms.forEach(a => {
        const card = document.createElement('div');
        card.className = 'algorithm-card';
        card.dataset.algoId = a.id;
        card.innerHTML = `<div class="algo-icon">${ALGO_ICONS[a.id] || '⚙️'}</div><div class="algo-name">${a.name}</div><div class="algo-desc">${a.description}</div>`;
        card.onclick = () => selectAlgorithm(a.id);
        grid.appendChild(card);
    });
}

function selectAlgorithm(id) {
    state.algorithm = id;
    document.querySelectorAll('.algorithm-card').forEach(c => c.classList.toggle('selected', c.dataset.algoId === id));
}

function initFileUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => handleFile(e.target.files[0]);
}

function handleFile(file) {
    if (!file) return;
    state.file = file;
    document.getElementById('drop-zone').style.display = 'none';
    document.getElementById('file-info').style.display = 'flex';
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('stat-dataset').textContent = file.name;
}

function clearFile() {
    state.file = null;
    document.getElementById('drop-zone').style.display = '';
    document.getElementById('file-info').style.display = 'none';
    document.getElementById('results-panel').style.display = 'none';
    document.getElementById('results-empty').style.display = '';
}

function initEvalMethodToggle() {
    document.querySelectorAll('input[name="eval-method"]').forEach(r => {
        r.onchange = () => {
            const isCV = r.value === 'crossvalidation';
            document.getElementById('cv-settings').style.display = isCV ? '' : 'none';
            document.getElementById('ps-settings').style.display = isCV ? 'none' : '';
        };
    });
}

async function runClassification() {
    if (!state.file || !state.algorithm) return alert("Select file and algorithm");
    
    const formData = new FormData();
    formData.append('file', state.file);
    formData.append('algorithm', state.algorithm);
    formData.append('evaluationMethod', document.querySelector('input[name="eval-method"]:checked').value);
    formData.append('folds', document.getElementById('folds').value);
    formData.append('trainPercent', document.getElementById('train-percent').value);

    document.getElementById('loading-overlay').style.display = 'flex';
    try {
        const resp = await fetch('/api/weka/classify', { method: 'POST', body: formData });
        const results = await resp.json();
        renderResults(results);
    } catch (e) {
        alert("Error: " + e.message);
    } finally {
        document.getElementById('loading-overlay').style.display = 'none';
    }
}

function renderResults(r) {
    document.getElementById('results-empty').style.display = 'none';
    document.getElementById('results-panel').style.display = '';
    document.getElementById('stat-instances').textContent = r.numInstances;
    document.getElementById('stat-accuracy').textContent = r.accuracy.toFixed(1) + '%';
    document.getElementById('accuracy-value').textContent = r.accuracy.toFixed(1) + '%';
    document.getElementById('detail-kappa').textContent = r.kappa.toFixed(4);
    document.getElementById('detail-evaluation').textContent = r.evaluationMethod;
    
    // Confusion Matrix
    renderCM(r.confusionMatrix, r.classNames);
}

function renderCM(matrix, names) {
    let html = '<table><tr><th></th>' + names.map(n => `<th>${n}</th>`).join('') + '</tr>';
    matrix.forEach((row, i) => {
        html += `<tr><th>${names[i]}</th>` + row.map(v => `<td>${v}</td>`).join('') + '</tr>';
    });
    html += '</table>';
    document.getElementById('confusion-matrix-container').innerHTML = html;
}
