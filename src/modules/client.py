import gradio as gr 
import pandas as pd 
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import random
from modules.finagent_system import build_master_workflow, MasterState
import json
from time import sleep


def fetch_stock_data(symbol, period = "1d", interval = "1m"): 


    try : 
        stock = yf.Ticker(symbol)
        data = stock.history(period = period, interval = interval)
        return data 
    
    except Exception as e: 
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()




def create_chart(symbol): 
    data = fetch_stock_data(symbol)
    if data.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        )
    )
    
    # Add volume as a bar chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.3)',
            yaxis='y2'
        )
    )
    
    # Add moving averages if we have enough data
    if len(data) >= 20:
        data['MA20'] = data['Close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['MA20'],
                line=dict(color='orange', width=1),
                name='20-day MA'
            )
        )
    
    if len(data) >= 50:
        data['MA50'] = data['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['MA50'],
                line=dict(color='green', width=1),
                name='50-day MA'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Price (Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(
            title="Volume",
            side="right",
            overlaying="y",
            showgrid=False
        ),
        height=600,
    )
    
    return fig 


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
            data = pd.read_csv(f"cache/{symbol}_data.csv")
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
        title="Portfolio Composition",
        height=300
    )
    
    return fig

def create_performance_chart():
    # Mock performance data - in a real app this would come from a database
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30).tolist()
    
    # Generate some mock portfolio performance data
    portfolio_value = [100000]  # Starting with $100,000
    for i in range(1, len(dates)):
        # Generate a random daily change between -2% and 3%
        daily_change = (random.random() * 5 - 2) / 100
        portfolio_value.append(portfolio_value[-1] * (1 + daily_change))
    
    # Market benchmark (e.g., S&P 500)
    benchmark = [100000]  # Starting at the same value
    for i in range(1, len(dates)):
        daily_change = (random.random() * 4 - 1.5) / 100
        benchmark.append(benchmark[-1] * (1 + daily_change))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=portfolio_value,
        mode='lines',
        name='Your Portfolio',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, 
        y=benchmark,
        mode='lines',
        name='Market Benchmark',
        line=dict(color='gray', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Performance (30 Days)",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=300
    )
    
    return fig

def create_profit_loss_chart():
    # Mock daily profit/loss data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=14).tolist()
    daily_pl = [(random.random() * 2000 - 800) for _ in range(len(dates))]
    
    colors = ['green' if val >= 0 else 'red' for val in daily_pl]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=daily_pl,
        marker_color=colors,
        name='Daily P/L'
    ))
    
    fig.update_layout(
        title="Daily Profit/Loss (Last 14 Days)",
        xaxis_title="Date",
        yaxis_title="Profit/Loss ($)",
        height=300
    )
    
    return fig

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
            <div class="decision-box" style="background-color: black; color: white;">
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
        return "Cache file not found", "", "", "", "", ""



    except Exception as e: 
        print(f"Error loading cached data for {symbol}: {e}")
        return None, None, None, None, None, None



def zenith_trader_app(): 
    with gr.Blocks(title="ZenithTrader", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 📈 ZenithTrader - AI-Powered Stock Analysis Platform
        
        Select a stock symbol, view real-time market data, and run AI trading analysis.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                stock_selector = gr.Dropdown(
                    choices=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                    value="AAPL", 
                    label="Select Stock Symbol",
                    interactive=True
                )
                
                with gr.Row():
                    period_selector = gr.Dropdown(
                        choices=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                        value="1d",
                        label="Period",
                        interactive=True
                    )
                    
                    interval_selector = gr.Dropdown(
                        choices=["1m", "5m", "15m", "30m", "60m", "1d"],
                        value="1m",
                        label="Interval",
                        interactive=True
                    )
            
            with gr.Column(scale=1):
                refresh_button = gr.Button("🔄 Refresh Chart")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Market Data"):
                with gr.Row():
                    stock_chart = gr.Plot(label="Stock Price Chart")
                
                with gr.Row():
                    run_button = gr.Button("🤖 Run Trading Analysis", variant="primary")
            
            with gr.TabItem("Portfolio Dashboard"):
                with gr.Row():
                    with gr.Column():
                        portfolio_pie = gr.Plot()
                    with gr.Column():
                        performance_chart = gr.Plot()
                
                with gr.Row():
                    profit_loss_chart = gr.Plot()

        # CSS for styling agent output
        gr.HTML("""
        <style>
        .decision-box {
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .action-buy {
            color: #4CAF50;
            font-weight: bold;
        }
        .action-sell {
            color: #F44336;
            font-weight: bold.
        }
        .action-hold {
            color: #2196F3;
            font-weight: bold.
        }
        </style>
        """)
        
        # Agent output area with tabs
        with gr.Accordion("Trading Analysis Results", open=False) as analysis_accordion:
            status_text = gr.Markdown("Select a stock and run analysis")
            
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
        
        # Update chart when stock is selected or refresh button is clicked
        stock_selector.change(
            fn=create_chart,
            inputs=stock_selector,
            outputs=stock_chart
        )
        
        refresh_button.click(
            fn=create_chart,
            inputs=stock_selector,
            outputs=stock_chart
        )
        
        # Run agent when button is clicked - fixed by removing _js parameter
        run_button.click(
            fn=run_agent_from_cache,
            inputs=[stock_selector, period_selector, interval_selector],
            outputs=[
                status_text,
                market_data_md,
                tech_analysis_md,
                strategy_md,
                decision_html,
                risk_md
            ]
        )
        
        # Initialize with default stock chart and portfolio dashboard
        app.load(
            fn=lambda: [create_chart("AAPL"), create_portfolio_chart(), create_performance_chart(), create_profit_loss_chart()],
            inputs=None,
            outputs=[stock_chart, portfolio_pie, performance_chart, profit_loss_chart]
        )
    
    return app




if __name__ == "__main__":
    


    app = zenith_trader_app()
    app.launch() 


