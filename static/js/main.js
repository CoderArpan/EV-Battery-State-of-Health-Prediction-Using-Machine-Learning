// Chart.js Global Settings for Dark Mode
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.legend.display = false;
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;

document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
});

async function initDashboard() {
    try {
        // Fetch all initial data in parallel
        const [
            metrics, 
            features, 
            predictions, 
            distribution, 
            statusDist,
            correlation,
            dataset, 
            options
        ] = await Promise.all([
            fetch('/api/metrics').then(r => r.json()),
            fetch('/api/feature-importance').then(r => r.json()),
            fetch('/api/predictions').then(r => r.json()),
            fetch('/api/distribution').then(r => r.json()),
            fetch('/api/status-distribution').then(r => r.json()),
            fetch('/api/correlation').then(r => r.json()),
            fetch('/api/dataset').then(r => r.json()),
            fetch('/api/form-options').then(r => r.json())
        ]);

        // Render UI Elements
        renderMetrics(metrics);
        renderGaugeStats(dataset.stats);
        
        // Render Charts
        renderScatterChart(predictions);
        renderFeatureChart(features);
        renderStatusDonut(statusDist);
        renderDistributionChart(distribution);
        renderCorrelationMatrix(correlation);
        
        // Form Setup
        populateForm(options);
        setupPredictionForm();
        
    } catch (error) {
        console.error("Failed to initialize dashboard:", error);
        document.getElementById('server-status').textContent = "Connection Error";
        document.querySelector('.pulse-dot').style.background = "var(--accent-red)";
    }
}

// ------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------
function animateValue(obj, start, end, duration, isPercentage = false) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
        const current = start + easeProgress * (end - start);
        
        if (isPercentage) {
            obj.innerHTML = current.toFixed(2) + '%';
        } else {
            obj.innerHTML = current.toFixed(4);
        }
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// ------------------------------------------------------------------
// Metrics & Stats
// ------------------------------------------------------------------
function renderMetrics(data) {
    animateValue(document.getElementById('metric-r2'), 0, data.r2, 1500);
    animateValue(document.getElementById('metric-mae'), 0, data.mae, 1500);
    animateValue(document.getElementById('metric-rmse'), 0, data.rmse, 1500);
    animateValue(document.getElementById('metric-accuracy'), 0, data.accuracy, 1500, true);
}

function renderGaugeStats(stats) {
    document.getElementById('stat-vehicles').textContent = stats.total_vehicles.toLocaleString();
    
    // Average SOH texts
    document.getElementById('stat-avg-soh').textContent = stats.avg_soh.toFixed(1) + '%';
    document.getElementById('stat-avg-soh-bottom').textContent = stats.avg_soh.toFixed(1) + '%';
    document.getElementById('stat-min-soh').textContent = stats.min_soh.toFixed(1) + '%';
    document.getElementById('stat-max-soh').textContent = stats.max_soh.toFixed(1) + '%';

    // Animate SVG Gauge
    const gaugeFill = document.getElementById('avg-gauge-fill');
    const circumference = 471.24; // 2 * pi * 75
    const offset = circumference - (stats.avg_soh / 100) * circumference;
    
    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = offset;
    }, 50);
}

// ------------------------------------------------------------------
// Charts
// ------------------------------------------------------------------
function renderScatterChart(data) {
    const ctx = document.getElementById('chart-scatter').getContext('2d');
    
    // Format data for scatter
    const points = data.actual.map((act, i) => ({
        x: act,
        y: data.predicted[i]
    }));

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                data: points,
                backgroundColor: 'rgba(0, 212, 170, 0.6)',
                borderColor: 'rgba(0, 212, 170, 0.9)',
                borderWidth: 1,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                // Perfect prediction reference line
                type: 'line',
                data: [{x: 70, y: 70}, {x: 100, y: 100}],
                borderColor: 'rgba(239, 68, 68, 0.5)',
                borderWidth: 1,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            scales: {
                x: {
                    title: { display: true, text: 'Actual SoH (%)', color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.02)' }
                },
                y: {
                    title: { display: true, text: 'Predicted SoH (%)', color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.02)' }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Actual: ${ctx.raw.x.toFixed(1)}%, Pred: ${ctx.raw.y.toFixed(1)}%`
                    }
                }
            }
        }
    });
}

function renderFeatureChart(data) {
    const ctx = document.getElementById('chart-features').getContext('2d');
    
    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, '#00d4aa'); // Teal
    gradient.addColorStop(1, '#f0b429'); // Amber

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.features.map(f => f.replace(/_/g, ' ')),
            datasets: [{
                data: data.importances,
                backgroundColor: gradient,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.02)' } },
                y: { grid: { display: false } }
            }
        }
    });
}

function renderStatusDonut(data) {
    const ctx = document.getElementById('chart-status').getContext('2d');
    
    // Healthy, Moderate, Critical
    const colors = ['#10b981', '#f0b429', '#ef4444'];
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.counts,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            cutout: '75%',
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: { color: '#64748b', boxWidth: 10, padding: 20 }
                }
            }
        }
    });
}

function renderDistributionChart(data) {
    const ctx = document.getElementById('chart-distribution').getContext('2d');
    
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(0, 212, 170, 0.8)');
    gradient.addColorStop(1, 'rgba(0, 212, 170, 0.0)');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.counts,
                backgroundColor: gradient,
                borderColor: '#00d4aa',
                borderWidth: 1,
                borderRadius: { topLeft: 4, topRight: 4 }
            }]
        },
        options: {
            scales: {
                x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45, font: {size: 9} } },
                y: { grid: { color: 'rgba(255,255,255,0.02)' } }
            }
        }
    });
}

// ------------------------------------------------------------------
// Correlation Matrix (HTML Table)
// ------------------------------------------------------------------
function renderCorrelationMatrix(data) {
    const table = document.getElementById('corr-table');
    const labels = data.labels;
    const matrix = data.matrix;
    
    // Header row (rotated)
    let thead = '<tr><th></th>';
    labels.forEach(l => {
        thead += `<th class="rot"><div>${l.replace(/_/g, ' ').toUpperCase()}</div></th>`;
    });
    thead += '</tr>';
    table.innerHTML = thead;

    // Body rows
    matrix.forEach((row, i) => {
        let tr = `<tr><td class="label-col">${labels[i].replace(/_/g, ' ')}</td>`;
        row.forEach(val => {
            // Color mapping: Negative=Red, Zero=Transparent, Positive=Teal
            let color = 'transparent';
            if (val > 0) {
                const alpha = Math.min(val, 1);
                color = `rgba(0, 212, 170, ${alpha})`;
            } else if (val < 0) {
                const alpha = Math.min(Math.abs(val), 1);
                color = `rgba(239, 68, 68, ${alpha})`;
            }
            if (val === 1.0) color = 'rgba(0, 212, 170, 1)'; // Solid for perfect corr

            tr += `<td class="corr-cell" style="background-color: ${color}">${val.toFixed(2)}</td>`;
        });
        tr += '</tr>';
        table.innerHTML += tr;
    });
}

// ------------------------------------------------------------------
// Form and Prediction Logic (Kept at bottom)
// ------------------------------------------------------------------
function populateForm(options) {
    const populateSelect = (id, items) => {
        const select = document.getElementById(id);
        items.forEach(item => {
            const opt = document.createElement('option');
            opt.value = item;
            opt.textContent = item;
            select.appendChild(opt);
        });
    };

    populateSelect('input-car-model', options.car_models);
    populateSelect('input-battery-type', options.battery_types);
    populateSelect('input-driving-style', options.driving_styles);
    
    document.getElementById('input-car-model').value = "Tesla Model 3";
    document.getElementById('input-driving-style').value = "Aggressive";
}

function setupPredictionForm() {
    const form = document.getElementById('prediction-form');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const btn = document.getElementById('predict-btn');
        btn.innerHTML = 'Analyzing...';
        btn.disabled = true;

        const payload = {
            "Car_Model": document.getElementById('input-car-model').value,
            "Battery_Type": document.getElementById('input-battery-type').value,
            "Battery_Capacity_kWh": parseFloat(document.getElementById('input-capacity').value),
            "Vehicle_Age_Months": parseFloat(document.getElementById('input-age').value),
            "Total_Charging_Cycles": parseFloat(document.getElementById('input-cycles').value),
            "Avg_Temperature_C": parseFloat(document.getElementById('input-temp').value),
            "Fast_Charge_Ratio": parseFloat(document.getElementById('input-fast-charge').value),
            "Avg_Discharge_Rate_C": parseFloat(document.getElementById('input-discharge').value),
            "Driving_Style": document.getElementById('input-driving-style').value,
            "Internal_Resistance_Ohm": parseFloat(document.getElementById('input-resistance').value)
        };

        try {
            const [predictRes, mechanicRes, whatIfRes] = await Promise.all([
                fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(r => r.json()),
                fetch('/api/mechanic-report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(r => r.json()),
                fetch('/api/prescriptive-analytics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(r => r.json())
            ]);

            document.getElementById('result-card').classList.remove('hidden');
            animateValue(document.getElementById('result-soh'), 0, predictRes.soh_prediction, 1000, true);
            
            const statusEl = document.getElementById('result-status');
            statusEl.textContent = predictRes.status;
            statusEl.className = 'status-badge large status-' + predictRes.status;
            
            document.getElementById('report-text').textContent = mechanicRes.report_text;
            
            const scenariosList = document.getElementById('scenarios-list');
            scenariosList.innerHTML = '';
            if (whatIfRes.scenarios && whatIfRes.scenarios.length > 0) {
                whatIfRes.scenarios.forEach(scen => {
                    scenariosList.innerHTML += `<div class="scenario-item"><span>${scen.action}</span> <strong style="color:var(--primary)">+${scen.improvement.toFixed(2)}%</strong></div>`;
                });
            }
            
        } catch (error) {
            console.error("Prediction failed:", error);
            alert("Error running diagnostics.");
        } finally {
            btn.innerHTML = 'Run Diagnostics';
            btn.disabled = false;
        }
    });
}
