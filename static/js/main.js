document.addEventListener('DOMContentLoaded', () => {
    // Navigation Tabs & Mobile Drawer
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    const sidebar = document.getElementById('sidebar');
    const mobileToggleBtn = document.getElementById('mobile-toggle-btn');
    const mobileCloseBtn = document.getElementById('mobile-close-btn');

    // Mobile Drawer Controls
    if (mobileToggleBtn) {
        mobileToggleBtn.addEventListener('click', () => sidebar.classList.add('mobile-open'));
    }
    if (mobileCloseBtn) {
        mobileCloseBtn.addEventListener('click', () => sidebar.classList.remove('mobile-open'));
    }

    const tabMeta = {
        'live': {
            title: 'Live Text & Tweet Analyzer',
            subtitle: 'Real-time Hybrid VADER Lexical & Naive Bayes ML Sentiment Evaluation'
        },
        'dataset': {
            title: 'Dataset Analytics & Exploratory Insights',
            subtitle: 'Statistical Breakdown of Indexed Social Media Tweets'
        },
        'batch': {
            title: 'Multi-Line Tweet Batch Evaluator',
            subtitle: 'Bulk Sentiment Diagnostics, Execution Metrics, and Data Export'
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

            // Close mobile menu when tab is clicked
            sidebar.classList.remove('mobile-open');

            if (targetTab === 'dataset' && !window.datasetLoaded) {
                fetchDatasetStats();
            }
        });
    });

    // Alert Notification Toast Helper
    window.showAlert = function(message, type = 'error') {
        const container = document.getElementById('alert-container');
        const messageEl = document.getElementById('alert-message');
        if (container && messageEl) {
            messageEl.textContent = message;
            container.className = `alert-container alert-${type}`;
            container.style.display = 'flex';
        }
    };

    window.closeAlert = function() {
        const container = document.getElementById('alert-container');
        if (container) container.style.display = 'none';
    };

    // Live Analyzer Controls & Character Counter
    const inputArea = document.getElementById('analyze-input');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnClear = document.getElementById('btn-clear');
    const charCountEl = document.getElementById('char-count');

    if (inputArea && charCountEl) {
        inputArea.addEventListener('input', () => {
            const len = inputArea.value.length;
            charCountEl.textContent = `${len} / 5000`;
        });
    }

    btnAnalyze.addEventListener('click', runLiveAnalysis);
    btnClear.addEventListener('click', () => {
        inputArea.value = '';
        if (charCountEl) charCountEl.textContent = '0 / 5000';
        resetLiveOutputs();
        closeAlert();
    });

    // Sample Texts
    window.loadSample = function(type) {
        closeAlert();
        if (type === 'positive') {
            inputArea.value = "Borderlands is absolutely fantastic! The gameplay mechanics, visuals, and characters are top-tier perfection! 😍🔥";
        } else if (type === 'negative') {
            inputArea.value = "Extremely disappointed with this update. The game keeps freezing, servers are terrible, and support is non-existent. 😡";
        } else if (type === 'mixed') {
            inputArea.value = "The graphics are breathtaking and the story is great, but the multiplayer connection issues are really frustrating.";
        }
        if (charCountEl) charCountEl.textContent = `${inputArea.value.length} / 5000`;
        runLiveAnalysis();
    };

    async function runLiveAnalysis() {
        const text = inputArea.value.trim();
        if (!text) {
            showAlert('Please enter some text before analyzing.', 'warning');
            return;
        }

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error occurred during analysis.');
            }

            const data = await response.json();
            updateLiveResults(data);
            closeAlert();
        } catch (err) {
            console.error('Analysis Error:', err);
            showAlert(err.message || 'Failed to analyze sentiment. Please verify server connection.', 'error');
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
        const signalsContainer = document.getElementById('ml-signals-pills');
        const latencyBadge = document.getElementById('latency-badge');
        const resultLatency = document.getElementById('result-latency');

        // Latency
        const latencyText = `${data.processing_time_ms || 0} ms`;
        if (latencyBadge) latencyBadge.innerHTML = `<i class="fa-solid fa-bolt"></i> Latency: ${latencyText}`;
        if (resultLatency) resultLatency.textContent = latencyText;

        // Consensus Badge
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

        // ML Matched Feature Signal Pills
        if (signalsContainer) {
            signalsContainer.innerHTML = '';
            const matched = data.ml.matched_features || [];
            if (matched.length > 0) {
                matched.forEach(feature => {
                    const span = document.createElement('span');
                    span.className = 'signal-pill';
                    span.textContent = feature;
                    signalsContainer.appendChild(span);
                });
            } else {
                signalsContainer.innerHTML = '<span class="signal-empty">No specific feature unigrams matched</span>';
            }
        }
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
        document.getElementById('ml-signals-pills').innerHTML = '<span class="signal-empty">None detected yet</span>';
        document.getElementById('latency-badge').innerHTML = '<i class="fa-solid fa-bolt"></i> Latency: -- ms';
        document.getElementById('result-latency').textContent = '-- ms';
    }

    // Chart.js Dataset Analytics
    let globalChart = null;
    let entityChart = null;

    async function fetchDatasetStats() {
        try {
            const response = await fetch('/api/stats');
            if (!response.ok) throw new Error('Failed to fetch dataset statistics.');
            const data = await response.json();

            document.getElementById('stat-total-records').textContent = data.total_records.toLocaleString();
            document.getElementById('stat-unique-entities').textContent = data.unique_entities;
            document.getElementById('stat-agreement-rate').textContent = `${data.vader_ml_agreement_rate || 88.5}%`;

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

            select.onchange = (e) => fetchEntityDetails(e.target.value);
            if (data.top_entities.length > 0) {
                fetchEntityDetails(data.top_entities[0][0]);
            }

            window.datasetLoaded = true;
        } catch (err) {
            console.error('Dataset stats error:', err);
            showAlert('Could not load dataset analytics.', 'error');
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
                    legend: { position: 'bottom', labels: { color: '#9CA3AF', font: { family: 'Inter' } } }
                }
            }
        });
    }

    async function fetchEntityDetails(entityName) {
        const list = document.getElementById('entity-samples-list');
        list.innerHTML = '<li class="sample-loading">Loading representative tweet samples...</li>';

        try {
            const response = await fetch(`/api/entity/${encodeURIComponent(entityName)}`);
            if (!response.ok) throw new Error('Failed to load entity details');
            const data = await response.json();

            renderEntityChart(data.distribution);

            // Render sample list
            list.innerHTML = '';
            if (data.samples && data.samples.length > 0) {
                data.samples.forEach(sample => {
                    const li = document.createElement('li');
                    let badgeClass = sample.sentiment === 'Positive' ? 'badge-pos' : (sample.sentiment === 'Negative' ? 'badge-neg' : 'badge-neutral');
                    li.innerHTML = `
                        <span>"${sample.text}"</span>
                        <span class="badge ${badgeClass}">${sample.sentiment}</span>
                    `;
                    list.appendChild(li);
                });
            } else {
                list.innerHTML = '<li>No tweet samples available for this entity.</li>';
            }
        } catch (err) {
            console.error('Entity details error:', err);
            list.innerHTML = '<li class="text-error">Error loading tweet samples.</li>';
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
    const btnBatchClear = document.getElementById('btn-batch-clear');
    const batchInput = document.getElementById('batch-input');
    const batchResultsWrapper = document.getElementById('batch-results-wrapper');
    const batchTableBody = document.getElementById('batch-table-body');
    const btnExportCSV = document.getElementById('btn-export-csv');
    const btnExportJSON = document.getElementById('btn-export-json');

    let currentBatchResults = [];

    if (btnBatchClear) {
        btnBatchClear.addEventListener('click', () => {
            batchInput.value = '';
            batchResultsWrapper.style.display = 'none';
            currentBatchResults = [];
        });
    }

    btnBatch.addEventListener('click', async () => {
        const lines = batchInput.value.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        if (lines.length === 0) {
            showAlert('Please enter at least one line of text for batch processing.', 'warning');
            return;
        }

        btnBatch.disabled = true;
        btnBatch.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Batch...';

        try {
            const response = await fetch('/api/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts: lines })
            });

            if (!response.ok) throw new Error('Batch processing request failed.');
            const data = await response.json();
            
            currentBatchResults = data.results || [];
            renderBatchResults(currentBatchResults);
            batchResultsWrapper.style.display = 'block';
            closeAlert();
        } catch (err) {
            console.error('Batch error:', err);
            showAlert('Batch processing failed. Check console for details.', 'error');
        } finally {
            btnBatch.disabled = false;
            btnBatch.innerHTML = '<i class="fa-solid fa-play"></i> Run Batch Diagnostics';
        }
    });

    function renderBatchResults(results) {
        batchTableBody.innerHTML = '';
        let posCount = 0, negCount = 0, neuCount = 0;

        results.forEach((res, index) => {
            if (res.consensus_label === 'Positive') posCount++;
            else if (res.consensus_label === 'Negative') negCount++;
            else neuCount++;

            const tr = document.createElement('tr');
            let badgeClass = res.consensus_label === 'Positive' ? 'badge-pos' : (res.consensus_label === 'Negative' ? 'badge-neg' : 'badge-neutral');
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td class="batch-text-cell">${res.text}</td>
                <td><span class="badge ${badgeClass}">${res.consensus_label}</span></td>
                <td><code>${res.vader.compound > 0 ? '+' : ''}${res.vader.compound.toFixed(4)}</code></td>
                <td>${res.ml.label} (${(res.ml.confidence * 100).toFixed(0)}%)</td>
                <td><span class="latency-cell">${res.processing_time_ms || 0} ms</span></td>
            `;
            batchTableBody.appendChild(tr);
        });

        // Summary Stats Update
        const total = results.length;
        document.getElementById('batch-stat-total').textContent = total;
        document.getElementById('batch-stat-pos').textContent = `${((posCount / total) * 100).toFixed(0)}%`;
        document.getElementById('batch-stat-neg').textContent = `${((negCount / total) * 100).toFixed(0)}%`;
        document.getElementById('batch-stat-neu').textContent = `${((neuCount / total) * 100).toFixed(0)}%`;
    }

    // CSV Export
    if (btnExportCSV) {
        btnExportCSV.addEventListener('click', () => {
            if (!currentBatchResults || currentBatchResults.length === 0) return;
            let csvContent = "data:text/csv;charset=utf-8,Index,Text,Consensus,VADER_Compound,ML_Label,ML_Confidence,Latency_ms\n";
            currentBatchResults.forEach((r, i) => {
                const escapedText = `"${r.text.replace(/"/g, '""')}"`;
                csvContent += `${i + 1},${escapedText},${r.consensus_label},${r.vader.compound},${r.ml.label},${r.ml.confidence},${r.processing_time_ms || 0}\n`;
            });
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `sentipulse_batch_results_${Date.now()}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    // JSON Export
    if (btnExportJSON) {
        btnExportJSON.addEventListener('click', () => {
            if (!currentBatchResults || currentBatchResults.length === 0) return;
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentBatchResults, null, 2));
            const link = document.createElement("a");
            link.setAttribute("href", dataStr);
            link.setAttribute("download", `sentipulse_batch_results_${Date.now()}.json`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }
});
