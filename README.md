# Monte Carlo Stock Price Simulator

An interactive Python GUI application that performs Monte Carlo simulations on any stock using real market data from Yahoo Finance. The application forecasts potential price movements from the current date until year-end based on historical volatility and multiple scenarios.

## Features

- **Real-Time Data Integration**: Fetches live stock data using yfinance API
- **Automatic Volatility Calculation**: Computes annualized volatility from historical data
- **S&P 500 Based Scenarios**: Three market scenarios reflecting broad market expectations:
  - **Bull Case**: Strong market growth (+18% annual)
  - **Moderate Case**: Historical S&P 500 average (+10% annual)
  - **Bear Case**: Market correction (-8% annual)
- **Dynamic Simulation**: Runs 100 Monte Carlo paths using Geometric Brownian Motion
- **Animated Visualization**: Watch the simulation paths build up in real-time
- **Comprehensive Statistics**: Displays mean, median, and percentile forecasts
- **1-Year Forecast**: Projects stock prices 252 trading days (1 year) into the future

## Installation

1. **Clone or download the files**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install matplotlib numpy yfinance pandas
   ```

## Usage

1. **Run the application**:
   ```bash
   python stock_monte_carlo_gui.py
   ```

2. **In the GUI**:
   - Enter a ticker symbol (e.g., AAPL, TSLA, MSFT, GOOGL)
   - Click **"Load Data"** to fetch historical data and calculate volatility
   - Review the loaded stock information
   - Click **"Run Simulation"** to start the Monte Carlo animation

3. **Interpret Results**:
   - Green paths: Bull scenario (optimistic)
   - Yellow/Orange paths: Moderate scenario (historical trend)
   - Red paths: Bear scenario (pessimistic)
   - Thick lines show the median forecast for each scenario
   - Statistics panel shows projected year-end prices and returns

## How It Works

### Data Collection
- Fetches 1 year of historical price data from Yahoo Finance
- Calculates daily log returns
- Computes annualized volatility (σ)

### Volatility Calculation
```
σ = std(log returns) × √252
```

### Monte Carlo Simulation
Uses Geometric Brownian Motion (GBM):
```
S(t+1) = S(t) × exp((μ - 0.5σ²)Δt + σ√Δt × Z)
```
Where:
- S(t) = Stock price at time t
- μ = Expected drift (scenario-dependent)
- σ = Volatility
- Δt = Time step (1/252 for daily)
- Z = Random normal variable

### Scenario Definitions
- **Bull Case**: +18% annual return (strong market growth, above-average S&P 500)
- **Moderate Case**: +10% annual return (historical S&P 500 average)
- **Bear Case**: -8% annual return (market correction or mild recession)

## Example Stocks to Try

- **Tech Giants**: AAPL (Apple), MSFT (Microsoft), GOOGL (Google)
- **EV & Auto**: TSLA (Tesla), F (Ford), GM (General Motors)
- **Finance**: JPM (JPMorgan), BAC (Bank of America), GS (Goldman Sachs)
- **Retail**: WMT (Walmart), AMZN (Amazon), TGT (Target)
- **Energy**: XOM (Exxon), CVX (Chevron), BP (BP)
- **ETFs**: SPY (S&P 500), QQQ (Nasdaq), IWM (Russell 2000)

## Statistics Explained

### Key Metrics
- **Mean**: Average of all simulated year-end prices
- **Median**: Middle value (50th percentile) - less affected by outliers
- **10th Percentile**: 10% of outcomes fall below this price
- **90th Percentile**: 90% of outcomes fall below this price
- **Return %**: Percentage change from current price

### Reading the Results
- **Wide range between scenarios?** → High uncertainty in stock direction
- **Narrow confidence intervals?** → Lower volatility, more predictable
- **High volatility %?** → Expect larger price swings

## Limitations & Disclaimers

⚠️ **Important Notes**:
- This is for **educational and analytical purposes only**
- **NOT financial advice** - do not use as sole basis for investment decisions
- Past volatility does not guarantee future performance
- Assumes log-normal distribution (real markets have fat tails)
- Does not account for: dividends, splits, market events, news, or regime changes
- Results are probabilistic, not deterministic

## Technical Details

### Assumptions
1. Stock returns follow a log-normal distribution
2. Volatility remains constant (historical average)
3. No transaction costs or slippage
4. Continuous trading (no gaps)
5. Efficient markets (no arbitrage)

### Parameters
- **Simulations**: 100 paths per scenario
- **Time Horizon**: Current date → December 31st (current year)
- **Frequency**: Daily (252 trading days/year)
- **Random Seed**: 42 (for reproducibility)

## Troubleshooting

### Common Issues

1. **"No data found for ticker"**
   - Check spelling of ticker symbol
   - Verify ticker exists on Yahoo Finance
   - Try adding exchange suffix (e.g., TICKER.L for London)

2. **"Module not found"**
   - Ensure all packages are installed: `pip install -r requirements.txt`

3. **Slow loading**
   - Network issue fetching data from Yahoo Finance
   - Try again or check internet connection

4. **Animation stutters**
   - Normal on slower machines
   - Reduce `num_simulations` in code if needed (line 86)

## Customization

You can modify the code to:
- Change number of simulations (line 86: `self.num_simulations`)
- Adjust animation speed (line 379: `interval=30`)
- Modify scenario drift rates (lines 104-106)
- Extend to multi-year forecasts
- Add more statistical measures

## Requirements

- Python 3.7+
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- yfinance >= 0.2.0
- pandas >= 1.3.0

## License

This project is open-source and available for educational use.

## Contributing

Feel free to fork, modify, and enhance! Some ideas:
- Add options pricing
- Include dividends
- Support multiple tickers simultaneously
- Export results to Excel/CSV
- Add technical indicators
- Include correlation analysis for portfolios

---

**Happy Simulating! 📈📉**
