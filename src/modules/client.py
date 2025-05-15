import gradio as gr 
import pandas as pd 
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from modules.finagent_system import build_master_workflow, MasterState


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




def run_agent(symbol, period, interval): 

    try:
        # Create initial state for the agent
        state = MasterState(symbol=symbol)
        
        # Show "Running analysis..." message while processing
        yield f"Running analysis for {symbol}... (This may take a few minutes)"
        
        graph = build_master_workflow()

        # Run the agent workflow
        result = graph.invoke(state)
        
        # Format the output to display relevant information
        output = f"## Trading Analysis for {symbol}\n\n"
        
        if result.get("market_data"):
            output += "### Market Intelligence\n"
            output += result["market_data"][:500] + "...\n\n"
        
        if result.get("technical_analysis"):
            output += "### Technical Analysis\n"
            output += result["technical_analysis"][:500] + "...\n\n"
        
        if result.get("trading_strategy"):
            output += "### Trading Strategy\n"
            output += result["trading_strategy"][:500] + "...\n\n"
        
        if result.get("decision"):
            decision = result["decision"]
            output += f"### Decision: {decision.action}\n"
            output += f"Symbol: {decision.symbol}\n"
            output += f"Quantity: {decision.quantity}\n"
            output += f"Price: {decision.price}\n"
            output += f"Reason: {decision.reasoning}\n\n"
        
        if result.get("risk_assessment"):
            output += "### Risk Assessment\n"
            output += result["risk_assessment"][:500] + "...\n\n"
        
        yield output

    
    except Exception as e:
        print(f"Error running agent for {symbol}: {e}")
        yield f"Error running analysis for {symbol}: {e}"
   




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
            
            with gr.Column(scale=1):
                refresh_button = gr.Button("🔄 Refresh Chart")
        
        with gr.Row():
            stock_chart = gr.Plot(label="Stock Price Chart")
        
        with gr.Row():
            run_button = gr.Button("🤖 Run Trading Analysis", variant="primary")
        
        with gr.Row():
            output_box = gr.Markdown(label="Analysis Results")
        
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
        
        # Run agent when button is clicked
        run_button.click(
            fn=run_agent,
            inputs=stock_selector,
            outputs=output_box
        )
        
        # Initialize with default stock chart
        app.load(
            fn=lambda: create_chart("AAPL"),
            inputs=None,
            outputs=stock_chart
        )
    
    return app




if __name__ == "__main__":
    # Initialize the Gradio app
    app = zenith_trader_app()
    app.queue()
    app.launch(share = True) 


