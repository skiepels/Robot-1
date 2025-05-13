// dashboard/static/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const connectButton = document.getElementById('connectButton');
    const connectionStatus = document.getElementById('connectionStatus');
    const addStockButton = document.getElementById('addStockButton');
    const stockSymbolInput = document.getElementById('stockSymbol');
    const conditionsTable = document.getElementById('conditionsTable').querySelector('tbody');
    const patternsContainer = document.getElementById('patternsContainer');
    const tradesContainer = document.getElementById('tradesContainer');
    
    // Connect to Interactive Brokers
    connectButton.addEventListener('click', function() {
        fetch('/connect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                connectionStatus.textContent = 'Connected';
                connectionStatus.className = 'ms-3 badge bg-success';
                connectButton.disabled = true;
            } else {
                alert('Failed to connect: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error connecting to IB:', error);
            alert('Error connecting to Interactive Brokers');
        });
    });
    
    // Add stock to watchlist
    addStockButton.addEventListener('click', function() {
        const symbol = stockSymbolInput.value.trim().toUpperCase();
        if (symbol) {
            fetch('/add_stock', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symbol: symbol })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    stockSymbolInput.value = '';
                    updateDashboard();
                } else {
                    alert('Failed to add stock: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error adding stock:', error);
            });
        }
    });
    
    // Update dashboard data periodically
    function updateDashboard() {
        fetch('/stocks')
            .then(response => response.json())
            .then(data => {
                updateConnectionStatus(data.connection);
                updateConditionsTable(data.stocks);
                updatePatternsView(data.patterns);
                updateTradesView(data.trades);
            })
            .catch(error => {
                console.error('Error updating dashboard:', error);
            });
    }
    
    function updateConnectionStatus(status) {
        if (status.connected) {
            connectionStatus.textContent = 'Connected';
            connectionStatus.className = 'ms-3 badge bg-success';
            connectButton.disabled = true;
        } else {
            connectionStatus.textContent = status.message || 'Not Connected';
            connectionStatus.className = 'ms-3 badge bg-danger';
            connectButton.disabled = false;
        }
    }
    
    function updateConditionsTable(stocks) {
        conditionsTable.innerHTML = '';
        
        Object.values(stocks).forEach(stock => {
            const row = document.createElement('tr');
            
            // Create status cells for each condition
            const priceCell = createConditionCell(stock.conditions.price, `$${stock.price?.toFixed(2)}`);
            const gapCell = createConditionCell(stock.conditions.percent_up, `${stock.gap_percent?.toFixed(2)}%`);
            const volumeCell = createConditionCell(stock.conditions.volume, `${stock.rel_volume?.toFixed(2)}x`);
            const newsCell = createConditionCell(stock.conditions.news, stock.has_news ? 'Yes' : 'No');
            const floatCell = createConditionCell(stock.conditions.float, `${(stock.float / 1000000).toFixed(1)}M`);
            
            // Overall status
            const statusCell = document.createElement('td');
            if (stock.all_conditions_met) {
                statusCell.innerHTML = '<span class="badge bg-success">All Met</span>';
            } else {
                statusCell.innerHTML = '<span class="badge bg-warning">Not All Met</span>';
            }
            
            // Build the row
            row.innerHTML = `<td>${stock.symbol}</td>`;
            row.appendChild(priceCell);
            row.appendChild(gapCell);
            row.appendChild(volumeCell);
            row.appendChild(newsCell);
            row.appendChild(floatCell);
            row.appendChild(statusCell);
            row.innerHTML += `
                <td>$${stock.price?.toFixed(2)}</td>
                <td>
                    <button class="btn btn-sm btn-danger remove-stock" data-symbol="${stock.symbol}">
                        <i class="bi bi-trash"></i>
                    </button>
                </td>
            `;
            
            conditionsTable.appendChild(row);
        });
        
        // Add event listeners to remove buttons
        document.querySelectorAll('.remove-stock').forEach(button => {
            button.addEventListener('click', function() {
                removeStock(this.getAttribute('data-symbol'));
            });
        });
    }
    
    function createConditionCell(condition, value) {
        const cell = document.createElement('td');
        const conditionMet = condition === true;
        
        cell.innerHTML = `
            <div class="condition-box ${conditionMet ? 'condition-met' : 'condition-not-met'}">
                ${value} ${conditionMet ? '<i class="bi bi-check"></i>' : '<i class="bi bi-x"></i>'}
            </div>
        `;
        
        return cell;
    }
    
    function updatePatternsView(patterns) {
        if (Object.keys(patterns).length === 0) {
            patternsContainer.innerHTML = '<div class="alert alert-info">No patterns detected yet</div>';
            return;
        }
        
        patternsContainer.innerHTML = '';
        
        Object.entries(patterns).forEach(([symbol, symbolPatterns]) => {
            symbolPatterns.forEach(pattern => {
                const patternCard = document.createElement('div');
                patternCard.className = 'pattern-card';
                
                patternCard.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <h4>${symbol} - ${pattern.pattern}</h4>
                        <span class="badge bg-primary">${pattern.confidence}% Confidence</span>
                    </div>
                    <div class="pattern-confidence">
                        <div class="pattern-confidence-fill" style="width: ${pattern.confidence}%"></div>
                    </div>
                    <div class="mt-3">
                        <div class="row">
                            <div class="col-md-4">
                                <strong>Entry:</strong> $${pattern.entry_price.toFixed(2)}
                            </div>
                            <div class="col-md-4">
                                <strong>Stop:</strong> $${pattern.stop_price.toFixed(2)}
                            </div>
                            <div class="col-md-4">
                                <strong>Target:</strong> $${pattern.target_price.toFixed(2)}
                            </div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <button class="btn btn-success trade-now" data-symbol="${symbol}" data-pattern="${pattern.pattern}">
                            Trade Now
                        </button>
                    </div>
                `;
                
                patternsContainer.appendChild(patternCard);
            });
        });
        
        // Add event listeners for trade buttons
        document.querySelectorAll('.trade-now').forEach(button => {
            button.addEventListener('click', function() {
                executeTrade(this.getAttribute('data-symbol'), this.getAttribute('data-pattern'));
            });
        });
    }
    
    function updateTradesView(trades) {
        if (Object.keys(trades).length === 0) {
            tradesContainer.innerHTML = '<div class="alert alert-info">No active trades</div>';
            return;
        }
        
        tradesContainer.innerHTML = '';
        
        Object.values(trades).forEach(trade => {
            const isProfitable = trade.unrealized_pnl > 0;
            const tradeCard = document.createElement('div');
            tradeCard.className = `trade-card ${isProfitable ? 'winning-trade' : 'losing-trade'}`;
            
            tradeCard.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <h4>${trade.symbol} - ${trade.pattern}</h4>
                    <span class="badge ${isProfitable ? 'bg-success' : 'bg-danger'}">
                        ${isProfitable ? '+' : ''}$${trade.unrealized_pnl?.toFixed(2)}
                    </span>
                </div>
                <div class="row mt-3">
                    <div class="col-md-3">
                        <strong>Entry:</strong> $${trade.entry_price.toFixed(2)}
                    </div>
                    <div class="col-md-3">
                        <strong>Current:</strong> $${trade.current_price?.toFixed(2)}
                    </div>
                    <div class="col-md-3">
                        <strong>Stop:</strong> $${trade.stop_price.toFixed(2)}
                    </div>
                    <div class="col-md-3">
                        <strong>Target:</strong> $${trade.target_price.toFixed(2)}
                    </div>
                </div>
                <div class="mt-3">
                    <button class="btn btn-warning btn-sm adjust-stop" data-symbol="${trade.symbol}">
                        Adjust Stop
                    </button>
                    <button class="btn btn-danger btn-sm exit-trade" data-symbol="${trade.symbol}">
                        Exit Trade
                    </button>
                </div>
            `;
            
            tradesContainer.appendChild(tradeCard);
        });
        
        // Add event listeners for trade management buttons
        document.querySelectorAll('.adjust-stop').forEach(button => {
            button.addEventListener('click', function() {
                adjustStop(this.getAttribute('data-symbol'));
            });
        });
        
        document.querySelectorAll('.exit-trade').forEach(button => {
            button.addEventListener('click', function() {
                exitTrade(this.getAttribute('data-symbol'));
            });
        });
    }
    
    function removeStock(symbol) {
        // Implement stock removal
        console.log('Removing stock:', symbol);
    }
    
    function executeTrade(symbol, pattern) {
        // Implement trade execution
        console.log('Executing trade:', symbol, pattern);
    }
    
    function adjustStop(symbol) {
        // Implement stop adjustment
        console.log('Adjusting stop for:', symbol);
    }
    
    function exitTrade(symbol) {
        // Implement trade exit
        console.log('Exiting trade:', symbol);
    }
    
    // Start periodic updates
    updateDashboard();
    setInterval(updateDashboard, 5000);
});