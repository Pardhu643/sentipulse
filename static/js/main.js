document.addEventListener('DOMContentLoaded', () => {
    // Navigation Tabs
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    const tabMeta = {
        'live': {
            title: 'Live Text & Tweet Analyzer',
            subtitle: 'Real-time Hybrid VADER Lexical & Naive Bayes ML Sentiment Evaluation'
        },
        'dataset': {
            title: 'Dataset Analytics & Exploratory Insights',
            subtitle: 'Statistical Breakdown of 74k+ Social Media Tweets'
        },
        'batch': {
            title: 'Multi-Line Tweet Batch Evaluator',
            subtitle: 'Bulk Sentiment Diagnostics and Metric Export'
        },
        'api': {
            title: 'API & System Architecture',
            subtitle: 'RESTful Endpoints & Hybrid NLP Execution Flow'
        }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.dataset.tab;

            navItems.forEach(i => i.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');

            if (tabMeta[targetTab]) {
                pageTitle.textContent = tabMeta[targetTab].title;
                pageSubtitle.textContent = tabMeta[targetTab].subtitle;
            }

            if (targetTab === 'dataset' && !window.datasetLoaded) {
                fetchDatasetStats();
            }
        });
    });

    // Live Analyzer Controls
    const inputArea = document.getElementById('analyze-input');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnClear = document.getElementById('btn-clear');

    btnAnalyze.addEventListener('click', runLiveAnalysis);
    btnClear.addEventListener('click', () => {
        inputArea.value = '';
        resetLiveOutputs();
    });

    // Sample Texts
    window.loadSample = function(type) {
        if (type === 'positive') {
            inputArea.value = "Borderlands is absolutely fantastic! The gameplay mechanics, visuals, and characters are top-tier perfection! 😍🔥";
        } else if (type === 'negative') {
            inputArea.value = "Extremely disappointed with this update. The game keeps freezing, servers are terrible, and support is non-existent. 😡";
        } else if (type === 'mixed') {
            inputArea.value = "The graphics are breathtaking and the story is great, but the multiplayer connection issues are really frustrating.";
        }
        runLiveAnalysis();
    };

    async function runLiveAnalysis() {
        const text = inputArea.value.trim();
        if (!text) return;

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();
            updateLiveResults(data);
        } catch (err) {
            console.error(err);
            alert('Failed to analyze sentiment. Please try again.');
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-microchip"></i> Analyze Sentiment';
        }
    }

    function updateLiveResults(data) {
        const consensusBadge = document.getElementById('consensus-badge');
        const compoundVal = document.getElementById('compound-val');
        const compoundFill = document.getElementById('compound-fill');
        const valPos = document.getElementById('val-pos');
        const valNeu = document.getElementById('val-neu');
        const valNeg = document.getElementById('val-neg');
        const mlLabel = document.getElementById('ml-label');
        const mlConfidence = document.getElementById('ml-confidence');

        // Badge update
        consensusBadge.textContent = data.consensus_label;
        consensusBadge.className = 'badge ';
        if (data.consensus_label === 'Positive') consensusBadge.classList.add('badge-pos');
        else if (data.consensus_label === 'Negative') consensusBadge.classList.add('badge-neg');
        else consensusBadge.classList.add('badge-neutral');

        // Compound Meter
        const compound = data.vader.compound;
        compoundVal.textContent = compound > 0 ? `+${compound.toFixed(4)}` : compound.toFixed(4);

        // Normalize compound (-1.0 to +1.0) -> (0% to 100%)
        const fillPercent = ((compound + 1) / 2) * 100;
        compoundFill.style.width = `${fillPercent}%`;

        if (compound >= 0.05) compoundFill.style.backgroundColor = 'var(--positive)';
        else if (compound <= -0.05) compoundFill.style.backgroundColor = 'var(--negative)';
        else compoundFill.style.backgroundColor = 'var(--neutral)';

        // Scores
        valPos.textContent = data.vader.pos.toFixed(2);
        valNeu.textContent = data.vader.neu.toFixed(2);
        valNeg.textContent = data.vader.neg.toFixed(2);

        // ML info
        mlLabel.textContent = data.ml.label;
        mlConfidence.textContent = `${(data.ml.confidence * 100).toFixed(1)}%`;
    }

    function resetLiveOutputs() {
        document.getElementById('consensus-badge').textContent = 'Awaiting Input';
        document.getElementById('consensus-badge').className = 'badge badge-neutral';
        document.getElementById('compound-val').textContent = '0.0000';
        document.getElementById('compound-fill').style.width = '50%';
        document.getElementById('compound-fill').style.backgroundColor = 'var(--primary)';
        document.getElementById('val-pos').textContent = '0.00';
        document.getElementById('val-neu').textContent = '0.00';
        document.getElementById('val-neg').textContent = '0.00';
        document.getElementById('ml-label').textContent = '--';
        document.getElementById('ml-confidence').textContent = '--';
    }

    // Chart.js Dataset Analytics
    let globalChart = null;
    let entityChart = null;

    async function fetchDatasetStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();

            document.getElementById('stat-total-records').textContent = data.total_records.toLocaleString();
            document.getElementById('stat-unique-entities').textContent = data.unique_entities;

            renderGlobalChart(data.sentiment_distribution);
            
            // Populate select options
            const select = document.getElementById('entity-select');
            select.innerHTML = '';
            data.top_entities.forEach(([entity, count]) => {
                const opt = document.createElement('option');
                opt.value = entity;
                opt.textContent = `${entity} (${count} tweets)`;
                select.appendChild(opt);
            });

            select.addEventListener('change', (e) => fetchEntityDetails(e.target.value));
            if (data.top_entities.length > 0) {
                fetchEntityDetails(data.top_entities[0][0]);
            }

            window.datasetLoaded = true;
        } catch (err) {
            console.error('Dataset stats error:', err);
        }
    }

    function renderGlobalChart(dist) {
        const ctx = document.getElementById('chart-global-sentiment').getContext('2d');
        if (globalChart) globalChart.destroy();

        globalChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(dist),
                datasets: [{
                    data: Object.values(dist),
                    backgroundColor: ['#10B981', '#EF4444', '#F59E0B', '#6366F1'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#9CA3AF' } }
                }
            }
        });
    }

    async function fetchEntityDetails(entityName) {
        try {
            const response = await fetch(`/api/entity/${encodeURIComponent(entityName)}`);
            const data = await response.json();

            renderEntityChart(data.distribution);

            // Render samples
            const list = document.getElementById('entity-samples-list');
            list.innerHTML = '';
            data.samples.forEach(sample => {
                const li = document.createElement('li');
                let badgeClass = sample.sentiment === 'Positive' ? 'badge-pos' : (sample.sentiment === 'Negative' ? 'badge-neg' : 'badge-neutral');
                li.innerHTML = `
                    <span>"${sample.text}"</span>
                    <span class="badge ${badgeClass}">${sample.sentiment}</span>
                `;
                list.appendChild(li);
            });
        } catch (err) {
            console.error('Entity details error:', err);
        }
    }

    function renderEntityChart(dist) {
        const ctx = document.getElementById('chart-entity-sentiment').getContext('2d');
        if (entityChart) entityChart.destroy();

        entityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(dist),
                datasets: [{
                    label: 'Tweet Count',
                    data: Object.values(dist),
                    backgroundColor: ['#10B981', '#EF4444', '#F59E0B', '#6366F1'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#9CA3AF' }, grid: { display: false } },
                    y: { ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // Batch Evaluation
    const btnBatch = document.getElementById('btn-batch-analyze');
    const batchInput = document.getElementById('batch-input');
    const batchResultsWrapper = document.getElementById('batch-results-wrapper');
    const batchTableBody = document.getElementById('batch-table-body');

    btnBatch.addEventListener('click', async () => {
        const lines = batchInput.value.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        if (lines.length === 0) return;

        btnBatch.disabled = true;
        btnBatch.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Batch...';

        try {
            const response = await fetch('/api/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts: lines })
            });

            const data = await response.json();
            batchTableBody.innerHTML = '';

            data.results.forEach((res, index) => {
                const tr = document.createElement('tr');
                let badgeClass = res.consensus_label === 'Positive' ? 'badge-pos' : (res.consensus_label === 'Negative' ? 'badge-neg' : 'badge-neutral');
                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${res.text}</td>
                    <td><span class="badge ${badgeClass}">${res.consensus_label}</span></td>
                    <td><code>${res.vader.compound > 0 ? '+' : ''}${res.vader.compound.toFixed(4)}</code></td>
                    <td>${res.ml.label} (${(res.ml.confidence * 100).toFixed(0)}%)</td>
                `;
                batchTableBody.appendChild(tr);
            });

            batchResultsWrapper.style.display = 'block';
        } catch (err) {
            console.error('Batch error:', err);
            alert('Batch processing failed.');
        } finally {
            btnBatch.disabled = false;
            btnBatch.innerHTML = '<i class="fa-solid fa-play"></i> Run Batch Diagnostics';
        }
    });
});
