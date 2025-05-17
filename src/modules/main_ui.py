import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime, timedelta
import yfinance as yf
import json
from time import sleep

from modules.finagent_system import MasterState, build_master_workflow

# Add fetch_stock_data function from client.py
def fetch_stock_data(symbol, period="1mo", interval="1d"): 
    try: 
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        return data 
    except Exception as e: 
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Create candlestick chart with real data
def create_candlestick_chart(symbol="BTC-USD", period="1mo", interval="1d"):
    df = fetch_stock_data(symbol, period, interval)
    if (df.empty):
        # Fallback to generated data if fetch fails
        df = generate_candlestick_data()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.02)
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index if hasattr(df, 'index') else df['Date'],
            open=df['Open'], 
            high=df['High'],
            low=df['Low'], 
            close=df['Close'],
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df.index if hasattr(df, 'index') else df['Date'], 
            y=df['Volume'],
            marker_color='rgba(73, 133, 231, 0.5)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'{symbol} Market Analysis',
            'x': 0.01,
            'xanchor': 'left'
        },
        xaxis_title='',
        yaxis_title='',
        template='plotly_dark',
        plot_bgcolor='rgba(15, 21, 42, 1)',
        paper_bgcolor='rgba(15, 21, 42, 1)',
        font=dict(color='white'),
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=40, r=20, t=50, b=20),
    )
    
    fig.update_xaxes(gridcolor='rgba(128, 128, 128, 0.1)', zerolinecolor='rgba(128, 128, 128, 0.1)')
    fig.update_yaxes(gridcolor='rgba(128, 128, 128, 0.1)', zerolinecolor='rgba(128, 128, 128, 0.1)')
    
    return fig

# Create portfolio chart with real data
def create_portfolio_chart():
    # Mock portfolio data - in a real app this would come from a database
    portfolio = {
        'AAPL': 10,
        'GOOGL': 5,
        'MSFT': 8,
        'AMZN': 3,
        'TSLA': 15
    }
    
    # Get current prices
    prices = {}
    total_value = 0
    for symbol, quantity in portfolio.items():
        try:
            data = fetch_stock_data(symbol, period="1d", interval="1d")
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                prices[symbol] = latest_price
                total_value += latest_price * quantity
        except Exception:
            prices[symbol] = 0
    
    # Create pie chart for portfolio composition
    labels = list(portfolio.keys())
    values = [portfolio[symbol] * prices.get(symbol, 0) for symbol in labels]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(
        title={
            'text': 'Portfolio Composition',
            'x': 0.01,
            'xanchor': 'left'
        },
        template='plotly_dark',
        plot_bgcolor='rgba(15, 21, 42, 1)',
        paper_bgcolor='rgba(15, 21, 42, 1)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=40, r=20, t=50, b=20),
    )
    
    return fig

# Create profit chart with real data for SPY (S&P 500 ETF) as benchmark
def create_profit_chart():
    try:
        # Get S&P 500 data for last 4 months
        data = fetch_stock_data('SPY', period="4mo", interval="1mo")
        months = []
        values = []
        
        if not data.empty:
            for i, (date, row) in enumerate(data.iterrows()):
                month = date.strftime('%b')
                months.append(month)
                # Calculate month-over-month change
                price_change = row['Close'] - row['Open']
                values.append(price_change)
        else:
            months = ['Jan', 'Feb', 'Mar', 'Apr']
            values = [random.uniform(100, 300) for _ in range(3)] + [random.uniform(500, 900)]
    except:
        months = ['Jan', 'Feb', 'Mar', 'Apr']
        values = [random.uniform(100, 300) for _ in range(3)] + [random.uniform(500, 900)]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=months,
            y=values,
            marker_color=['rgba(73, 133, 231, 0.7)' if v >= 0 else 'rgba(231, 73, 73, 0.7)' for v in values]
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Monthly Market Change',
            'x': 0.01,
            'xanchor': 'left'
        },
        xaxis_title='',
        yaxis_title='',
        template='plotly_dark',
        plot_bgcolor='rgba(15, 21, 42, 1)',
        paper_bgcolor='rgba(15, 21, 42, 1)',
        font=dict(color='white'),
        height=250,
        margin=dict(l=40, r=20, t=50, b=20),
        bargap=0.4
    )
    
    fig.update_xaxes(gridcolor='rgba(128, 128, 128, 0.1)', zerolinecolor='rgba(128, 128, 128, 0.1)')
    fig.update_yaxes(gridcolor='rgba(128, 128, 128, 0.1)', zerolinecolor='rgba(128, 128, 128, 0.1)')
    
    return fig

# Create performance chart with real data
def create_performance_chart(symbol='BTC-USD'):
    try:
        df = fetch_stock_data(symbol, period="1mo", interval="1d")
        if df.empty:
            raise ValueError("No data fetched")
            
        x_data = df.index
        y_data = df['Close']
        
    except Exception:
        # Fallback to generated data
        df = generate_price_data()
        x_data = df['Date']
        y_data = df['Price']
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            line=dict(color='#ffc400', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 196, 0, 0.1)'
        )
    )
    
    fig.update_layout(
        title={
            'text': f'{symbol} Performance Trend',
            'x': 0.01,
            'xanchor': 'left'
        },
        xaxis_title='',
        yaxis_title='',
        template='plotly_dark',
        plot_bgcolor='rgba(15, 21, 42, 1)',
        paper_bgcolor='rgba(15, 21, 42, 1)',
        font=dict(color='white'),
        height=150,
        margin=dict(l=40, r=20, t=50, b=20),
    )
    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig

# Keep the generate_* functions as fallbacks
def generate_price_data(days=30, volatility=0.02):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    price = 50000
    prices = [price]
    volumes = []
    
    for i in range(1, days):
        change = np.random.normal(0, volatility)
        price = price * (1 + change)
        prices.append(price)
        volumes.append(abs(change) * price * random.randint(100, 1000))
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volume': volumes if len(volumes) == len(dates) else volumes + [volumes[-1]]
    })
    return df

def generate_candlestick_data(days=60):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    data = []
    price = 50000
    
    for i in range(days):
        open_price = price
        high_price = open_price * (1 + random.uniform(0, 0.03))
        low_price = open_price * (1 - random.uniform(0, 0.03))
        close_price = random.uniform(low_price, high_price)
        
        data.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': random.randint(100, 5000) * 1000
        })
        
        price = close_price
    
    return pd.DataFrame(data)

def generate_arbitrage_data():
    exchanges = ["Binance", "Coinbase", "Kraken", "Huobi"]
    pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT"]
    
    data = []
    
    for _ in range(15):
        pair = random.choice(pairs)
        buy_exchange = random.choice(exchanges)
        sell_exchange = random.choice([ex for ex in exchanges if ex != buy_exchange])
        
        buy_price = random.uniform(0.9, 1.0) * (50000 if "BTC" in pair else 2500 if "ETH" in pair else 300 if "BNB" in pair else 0.5)
        sell_price = buy_price * random.uniform(1.001, 1.008)
        profit_pct = ((sell_price - buy_price) / buy_price) * 100
        
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 60))
        
        data.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "Pair": pair,
            "Buy": buy_exchange,
            "Buy Price": buy_price,
            "Sell": sell_exchange,
            "Sell Price": sell_price,
            "Profit %": round(profit_pct, 4)
        })
    
    return pd.DataFrame(data)

def generate_portfolio_data():
    coins = ["Bitcoin (BTC)", "Ethereum (ETH)", "Binance Coin (BNB)", "Ripple (XRP)", "Cardano (ADA)"]
    portfolio = []
    
    total_value = random.uniform(8000, 12000)
    remaining = total_value
    
    for i, coin in enumerate(coins[:-1]):
        if i == len(coins) - 2:
            value = remaining
        else:
            value = remaining * random.uniform(0.1, 0.4)
            remaining -= value
        
        portfolio.append({
            "Coin": coin,
            "Value": value,
            "Change": random.uniform(-5, 15)
        })
    
    return pd.DataFrame(portfolio)

# Updated run_agent function to match client.py format
def run_agent(symbol, period, interval): 
    try:
        # Create initial state for the agent
        state = MasterState(symbol=symbol)
        
        # Show "Running analysis..." message while processing
        yield f"Running analysis for {symbol}... (This may take a few minutes)", "", "", "", "", ""
        
        graph = build_master_workflow()

        # Run the agent workflow
        result = graph.invoke(state)
        
        # Structure the output for different tabs
        market_data = result.get("market_data", "No market data available")
        tech_analysis = result.get("technical_analysis", "No technical analysis available")
        trading_strategy = result.get("trading_strategy", "No trading strategy available")
        risk_assessment = result.get("risk_assessment", "No risk assessment available")
        
        # Format decision data
        decision_text = ""
        if result.get("decision"):
            decision = result["decision"]
            decision_text = f"""
            <div class="decision-box">
                <h3>Trading Decision: <span class="action-{decision.action.lower()}">{decision.action}</span></h3>
                <p><strong>Symbol:</strong> {decision.symbol}</p>
                <p><strong>Quantity:</strong> {decision.quantity}</p>
                <p><strong>Price:</strong> ${decision.price}</p>
                <p><strong>Reasoning:</strong> {decision.reasoning}</p>
            </div>
            """
        
        # Return individual values instead of a dictionary
        status_text = f"Analysis completed for {symbol}"
        
        yield status_text, market_data, tech_analysis, trading_strategy, decision_text, risk_assessment

    except Exception as e:
        print(f"Error running agent for {symbol}: {e}")
        error_msg = f"Error running analysis for {symbol}: {e}"
        yield error_msg, "", "", "", "", ""

# Add run_agent_from_cache function from client.py
def run_agent_from_cache(symbol, period, interval):
    try: 
        data = json.load(open(f"cache/result.json"))

        yield f"\n\n  Running analysis for {symbol}... (This may take a few minutes)", "", "", "", "", ""
        sleep(10) 

        market_data = data.get("market_data", "No market data available")
        tech_analysis = data.get("technical_analysis", "No technical analysis available")
        trading_strategy = data.get("trading_strategy", "No trading strategy available")
        risk_assessment = data.get("risk_assessment", "No risk assessment available")
        decision_text = ""
        if data.get("decision"):
            decision = data["decision"]
            decision_text = f"""
            <div class="decision-box" style="background-color: black; color: white;">
                <h3>Trading Decision: <span class="action-{decision['action'].lower()}">{decision['action']}</span></h3>
                <p><strong>Symbol:</strong> {decision['symbol']}</p>
                <p><strong>Quantity:</strong> {decision['quantity']}</p>
                <p><strong>Price:</strong> ${decision['price']}</p>
                <p><strong>Reasoning:</strong> {decision['reasoning']}</p>
            </div>
            """

        status_text = f"Analysis completed for {symbol}"
        
        yield status_text, market_data, tech_analysis, trading_strategy, decision_text, risk_assessment

    except FileNotFoundError:
        print(f"Cache file not found for {symbol}. Please run the analysis first.")
        yield "Cache file not found", "", "", "", "", ""

    except Exception as e: 
        print(f"Error loading cached data for {symbol}: {e}")
        yield f"Error loading cached data: {e}", "", "", "", "", ""

# Define CSS
css = """
body {
    background-color: #0a1628 !important;
    color: white !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gradio-container {
    background: linear-gradient(45deg, #0a1628, #132d5e) !important;
}

h1, h2, h3, h4 {
    color: white !important;
    padding-left: 15px !important;
    margin-left: 5px !important;
}

.tabs {
    border-radius: 10px;
    overflow: hidden;
    padding-left: 15px;
}

.card {
    background-color: rgba(15, 21, 42, 0.97) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    padding: 20px;
    margin-bottom: 20px;
}

.card h3, .card h4 {
    padding-left: 15px !important;
    margin-bottom: 20px !important;
    position: relative;
}

.card h3::before, .card h4::before {
    content: "";
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 5px;
    height: 20px;
    background: linear-gradient(45deg, #4985e7, #38bdf8);
    border-radius: 3px;
}

.balance-card {
    background: linear-gradient(145deg, rgba(15, 21, 42, 0.9), rgba(15, 21, 42, 0.97)) !important;
    border-left: 4px solid #4985e7 !important;
}

.secondary-card {
    background: rgba(15, 21, 42, 0.95) !important;
}

.header {
    display: flex;
    align-items: center;
    background-color: rgba(15, 21, 42, 0.95);
    padding: 10px 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.logo {
    font-size: 24px;
    font-weight: bold;
    background: linear-gradient(45deg, #4985e7, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-right: 20px;
    margin-left: 10px;
}

.menu-item {
    margin: 0 15px;
    opacity: 0.7;
    transition: opacity 0.3s;
    font-size: 16px;
    cursor: pointer;
}

.menu-item:hover {
    opacity: 1;
}

.active {
    opacity: 1;
    position: relative;
}

.active::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 0;
    width: 20px;
    height: 3px;
    background: linear-gradient(45deg, #4985e7, #38bdf8);
    border-radius: 3px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 14px;
}

th {
    text-align: left;
    padding: 12px 15px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    color: #888;
    font-weight: normal;
}

td {
    padding: 12px 15px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.1);
}

.positive {
    color: #26A69A !important;
}

.negative {
    color: #EF5350 !important;
}

.profit-donut {
    position: relative;
    width: 150px;
    height: 150px;
    margin: 0 auto;
}

.stats-value {
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 5px;
    padding-left: 10px;
}

.stats-label {
    font-size: 14px;
    color: #888;
    padding-left: 10px;
}

.bot-card {
    background-color: rgba(15, 21, 42, 0.95) !important;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    border-left: 3px solid #4985e7;
}

.btn-primary {
    background: linear-gradient(45deg, #4985e7, #38bdf8) !important;
    border: none !important;
    color: white !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    margin-left: 15px !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(73, 133, 231, 0.3) !important;
}

.btn-outline {
    background: transparent !important;
    border: 1px solid #4985e7 !important;
    color: #4985e7 !important;
    margin-left: 15px !important;
}

.btn-outline:hover {
    background: rgba(73, 133, 231, 0.1) !important;
}

.tabs {
    display: flex;
    margin-bottom: 15px;
}

.tab {
    padding: 8px 15px;
    cursor: pointer;
    opacity: 0.7;
    border-bottom: 2px solid transparent;
    transition: all 0.3s;
}

.tab:hover {
    opacity: 1;
}

.tab.active {
    opacity: 1;
    border-bottom: 2px solid #4985e7;
}

.filter-section {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding-left: 15px;
}

/* Ensure markdown content has proper padding */
.prose {
    padding-left: 15px !important;
}

/* Add padding to dataframe headers */
.svelte-1gfkn6j {
    padding-left: 15px !important;
}
"""

# Build the dashboard
with gr.Blocks(title = "Zenith Trader", css=css) as app:
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="header">
                <div class="logo">
                    <span style="font-size: 24px;">🤖</span> Zenith Trader 
                </div>
               
            </div>
            """)
    
    # Tabs for switching between Dashboard and Arbitrage views
    with gr.Tabs() as tabs:
        # Dashboard Tab
        with gr.TabItem("Dashboard"):
            with gr.Row():
                # Overview statistics
                with gr.Column():
                    with gr.Group(elem_classes=["card", "balance-card"]):
                        gr.Markdown("### Overview")
                        with gr.Row():
                            with gr.Column():
                                gr.HTML(f"""
                                <div style="text-align: center;">
                                    <div class="stats-value">${round(random.uniform(8000, 12000), 2)}</div>
                                    <div class="stats-label">Total Balance</div>
                                </div>
                                """)
                            with gr.Column():
                                gr.HTML(f"""
                                <div style="text-align: center;">
                                    <div class="stats-value">${round(random.uniform(2000, 4000), 2)}</div>
                                    <div class="stats-label">Profit</div>
                                </div>
                                """)
                            with gr.Column():
                                gr.HTML(f"""
                                <div style="text-align: center;">
                                    <div class="stats-value">{round(random.uniform(5, 15), 1)}%</div>
                                    <div class="stats-label">ROI</div>
                                </div>
                                """)
                
                # Profit and Loss Chart
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Profit and Loss")
                        with gr.Row():
                            with gr.Column():
                                gr.HTML("""
                                <div class="tabs">
                                    <div class="tab active">Daily</div>
                                    <div class="tab">Weekly</div>
                                    <div class="tab">Monthly</div>
                                    <div class="tab">Quarterly</div>
                                </div>
                                """)
                        profit_chart = gr.Plot(create_profit_chart)
            
            with gr.Row():
                # My Coins on Exchanges
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### My Coins on Exchanges")
                        
                        coins_df = pd.DataFrame({
                            "Coin": ["Bitcoin (BTC)", "Ethereum (ETH)", "Binance Coin (BNB)", "Cardano (ADA)", "Solana (SOL)"],
                            "Holdings": ["0.15 BTC", "2.5 ETH", "10 BNB", "500 ADA", "15 SOL"],
                            "Value (USD)": ["$7,500.32", "$5,621.45", "$3,150.00", "$520.50", "$1,428.75"],
                            "Change": ["+4.2%", "+3.1%", "-1.5%", "+2.8%", "+9.5%"]
                        })
                        
                        gr.Dataframe(
                            coins_df,
                            headers=["Coin", "Holdings", "Value (USD)", "Change"],
                            elem_classes=["crypto-table"]
                        )
                
                # Active Bots
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Active Bots")
                        
                        bots_df = pd.DataFrame({
                            "Bot": ["BTC/ETH Arbitrage", "XRP Cross-Exchange", "Binance DCA"],
                            "Status": ["ACTIVE", "ACTIVE", "PAUSED"],
                            "Profit/Loss": ["+$251.32", "+$75.18", "+$124.50"]
                        })
                        
                        gr.Dataframe(
                            bots_df,
                            headers=["Bot", "Status", "Profit/Loss"],
                            elem_classes=["bots-table"]
                        )
                        
                        gr.Button("START MORE", elem_classes=["btn-primary"])
            
            # Top Cryptocurrencies
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Top Cryptocurrencies")
                        
                        with gr.Row():
                            with gr.Column():
                                gr.HTML("""
                                <div class="tabs">
                                    <div class="tab active">Trending</div>
                                    <div class="tab">Opportunities</div>
                                </div>
                                """)
                        
                        with gr.Row():
                            with gr.Column():
                                # Bitcoin performance
                                gr.Markdown("#### Bitcoin (BTC)")
                                gr.HTML(f"""<div><span class="stats-value">$48,532.12</span> <span class="positive">+1.2%</span></div>""")
                                btc_chart = gr.Plot(create_performance_chart)
                            
                            with gr.Column():
                                # Ethereum performance
                                gr.Markdown("#### Ethereum (ETH)")
                                gr.HTML(f"""<div><span class="stats-value">$2,254.85</span> <span class="positive">+3.5%</span></div>""")
                                eth_chart = gr.Plot(create_performance_chart)

        # Arbitrage Tab
        with gr.TabItem("Arbitrage Bots"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### AI Strategy")
                        symbol_select = gr.Dropdown(
                            choices=["BTC-USD", "ETH-USD", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                            value="BTC-USD",
                            label="Select Symbol"
                        )
                        candlestick = gr.Plot(lambda: create_candlestick_chart("BTC-USD"))
                        symbol_select.change(
                            fn=create_candlestick_chart,
                            inputs=symbol_select,
                            outputs=candlestick
                        )
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Strategies")
                        
                        gr.HTML("""
                        <div class="tabs">
                            <div class="tab active">DCA</div>
                            <div class="tab">Grid</div>
                            <div class="tab">MACD</div>
                            <div class="tab">Custom</div>
                        </div>
                        """)
                        
                        strategies_df = pd.DataFrame({
                            "Pair": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOT/USDT", "DOGE/USDT", "SOL/USDT"],
                            "Strategy": ["Long", "Long", "Long", "Short", "Long", "Long", "Short", "Long"],
                            "Profit": ["+5.2%", "+3.8%", "+1.2%", "-0.7%", "+4.5%", "+2.3%", "-1.5%", "+8.1%"]
                        })
                        
                        gr.Dataframe(
                            strategies_df,
                            headers=["Pair", "Strategy", "Profit"],
                            elem_classes=["strategies-table"]
                        )
            
            # New section - Agent Suggestions
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### AI Agent Suggestions")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                symbol_input = gr.Dropdown(
                                    choices=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "BTC-USD", "ETH-USD"],
                                    label="Symbol",
                                    value="AAPL"
                                )
                            with gr.Column(scale=1):
                                period_input = gr.Dropdown(
                                    choices=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                                    label="Period",
                                    value="3mo"
                                )
                            with gr.Column(scale=1):
                                interval_input = gr.Dropdown(
                                    choices=["1m", "5m", "15m", "30m", "1h", "1d"],
                                    label="Interval",
                                    value="1h"
                                )
                        
                        with gr.Row():
                            run_cache_button = gr.Button("Get AI Suggestions", elem_classes=["btn-outline"])
                        
                        # Agent output area with tabs
                        with gr.Accordion("Trading Analysis Results", open=False) as analysis_accordion:
                            status_text = gr.Markdown("Select a symbol and run analysis")
                            
                            with gr.Tabs() as analysis_tabs:
                                with gr.TabItem("Decision"):
                                    decision_html = gr.HTML("")
                                with gr.TabItem("Market Intelligence"):
                                    market_data_md = gr.Markdown("")
                                with gr.TabItem("Technical Analysis"):
                                    tech_analysis_md = gr.Markdown("")
                                with gr.TabItem("Trading Strategy"):
                                    strategy_md = gr.Markdown("")
                                with gr.TabItem("Risk Assessment"):
                                    risk_md = gr.Markdown("")
                        
                        # run_button.click(
                        #     fn=run_agent,
                        #     inputs=[symbol_input, period_input, interval_input],
                        #     outputs=[
                        #         status_text,
                        #         market_data_md,
                        #         tech_analysis_md,
                        #         strategy_md,
                        #         decision_html,
                        #         risk_md
                        #     ]
                        # )
                        
                        # Add the cached analysis button functionality
                        run_cache_button.click(
                            fn=run_agent_from_cache,
                            inputs=[symbol_input, period_input, interval_input],
                            outputs=[
                                status_text,
                                market_data_md,
                                tech_analysis_md,
                                strategy_md,
                                decision_html,
                                risk_md
                            ]
                        )
            
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Spot Bots")
                        
                        with gr.Row():
                            with gr.Column():
                                gr.HTML("""
                                <div class="filter-section">
                                    <div class="tabs">
                                        <div class="tab active">ALL</div>
                                        <div class="tab">GRID</div>
                                        <div class="tab">DCA</div>
                                    </div>
                                    <div>
                                        <button class="btn-primary">Create DCA Bot</button>
                                    </div>
                                </div>
                                """)
                        
                        # Arbitrage table
                        arb_df = generate_arbitrage_data()
                        gr.Dataframe(
                            arb_df,
                            headers=["Timestamp", "Pair", "Buy", "Buy Price", "Sell", "Sell Price", "Profit %"],
                            elem_classes=["arbitrage-table"]
                        )
            

# Launch the app
app.launch()