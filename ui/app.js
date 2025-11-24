let allData = null;
let currentChart = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('data.json');
        allData = await response.json();

        // Setup Tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Update Active State
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');

                // Switch Data
                const ticker = e.target.dataset.tab;
                switchTab(ticker);
            });
        });

        // Default to QQQ
        switchTab('qqq');

    } catch (error) {
        console.error('Error loading data:', error);
        document.body.innerHTML = '<div class="container"><h1>Error loading data. Please ensure data.json exists.</h1></div>';
    }
});

function switchTab(ticker) {
    if (!allData || !allData[ticker]) return;

    const data = allData[ticker];

    // Update Title
    document.getElementById('page-title').innerHTML = `${ticker.toUpperCase()} Strategy <span class="highlight">Performance</span>`;

    renderStats(data.stats);
    renderChart(data.equity);
    renderTrades(data.trades);
}

function renderStats(stats) {
    const grid = document.getElementById('stats-grid');

    const metrics = [
        { label: 'Total Return', value: `${stats.total_return.toFixed(2)}%`, color: stats.total_return >= 0 ? 'success-color' : 'danger-color' },
        { label: 'CAGR', value: `${(stats.cagr * 100).toFixed(2)}%` },
        { label: 'Win Rate', value: `${stats.win_rate.toFixed(2)}%` },
        { label: 'Total Trades', value: stats.total_trades },
        { label: 'Final Equity', value: `$${stats.final_equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
        { label: 'Buy & Hold', value: `${stats.bnh_return.toFixed(2)}%` }
    ];

    grid.innerHTML = metrics.map(m => `
        <div class="stat-card">
            <div class="stat-label">${m.label}</div>
            <div class="stat-value" style="${m.color ? `color: var(--${m.color})` : ''}">${m.value}</div>
        </div>
    `).join('');
}

function renderChart(equityData) {
    const ctx = document.getElementById('equityChart').getContext('2d');

    if (currentChart) {
        currentChart.destroy();
    }

    // Downsample if too many points for performance
    const points = equityData.length > 2000 ? equityData.filter((_, i) => i % 5 === 0) : equityData;

    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: points.map(d => d.date),
            datasets: [{
                label: 'Strategy Equity',
                data: points.map(d => d.equity),
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: '#334155',
                    borderWidth: 1,
                    callbacks: {
                        label: function (context) {
                            return '$' + context.parsed.y.toLocaleString();
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        maxTicksLimit: 8
                    }
                },
                y: {
                    grid: {
                        color: '#334155',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        callback: function (value) {
                            return '$' + value.toLocaleString(undefined, { notation: "compact" });
                        }
                    }
                }
            }
        }
    });
}

function renderTrades(trades) {
    const tbody = document.querySelector('#trades-table tbody');

    tbody.innerHTML = trades.map(t => {
        const isBuy = t.action.includes('BUY');
        const returnClass = t.return > 0 ? 'positive' : (t.return < 0 ? 'negative' : '');
        const returnText = t.return ? `${t.return > 0 ? '+' : ''}${(t.return * 100).toFixed(2)}%` : '-';

        return `
            <tr>
                <td>${t.date}</td>
                <td class="${isBuy ? 'action-buy' : 'action-sell'}">${t.action}</td>
                <td>$${t.price.toFixed(2)}</td>
                <td>${t.shares}</td>
                <td>$${t.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                <td class="${returnClass}">${returnText}</td>
            </tr>
        `;
    }).join('');
}
