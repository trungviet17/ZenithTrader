from modules.trading.finagent import FinAgent
from test.backtest import BacktestEngine
from data.data import create_sample_data

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FinAgent')


def main():
    """Demonstrate the usage of FinAgent."""
    # Create FinAgent instance
    agent = FinAgent()
    
    # Create BacktestEngine
    backtest_engine = BacktestEngine(agent)
    
    # Create sample data for testing
    sample_data = create_sample_data()
    
    # Run backtest
    results = backtest_engine.run_backtest(sample_data)
    
    # Print results
    print("Backtest Results:")
    print(f"Final Portfolio Value: ${results['metrics']['Final_Portfolio_Value']:.2f}")
    print(f"Total Return: {results['metrics']['Total_Return']:.2%}")
    print(f"Annual Rate of Return (ARR): {results['metrics']['ARR']:.2%}")
    print(f"Sharpe Ratio: {results['metrics']['Sharpe_Ratio']:.2f}")
    print(f"Maximum Drawdown: {results['metrics']['Max_Drawdown']:.2%}")
    print(f"Calmar Ratio: {results['metrics']['Calmar_Ratio']:.2f}")
    print(f"Sortino Ratio: {results['metrics']['Sortino_Ratio']:.2f}")
    print(f"Volatility: {results['metrics']['Volatility']:.2%}")
    
    print("\nTrades:")
    for trade in results['trades']:
        print(f"{trade['date']} - {trade['action']} {trade['shares']:.2f} shares at ${trade['price']:.2f} (${trade['value']:.2f})")


if __name__ == "__main__":
    main()