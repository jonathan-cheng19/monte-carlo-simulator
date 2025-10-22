import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockMonteCarloGUI:
    def __init__(self):
        self.setup_figure()
        self.ticker = None
        self.stock_data = None
        self.simulation_ready = False
        
    def setup_figure(self):
        """Initialize the GUI layout"""
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig = plt.figure(figsize=(18, 10), facecolor='#f5f5f5')
        self.gs = GridSpec(4, 1, figure=self.fig, height_ratios=[0.6, 4, 1.2, 0.5], hspace=0.35)
        
        # Input panel
        self.ax_input = self.fig.add_subplot(self.gs[0])
        self.ax_input.axis('off')
        
        # Main chart
        self.ax_chart = self.fig.add_subplot(self.gs[1])
        self.ax_chart.set_facecolor('#fafafa')
        
        # Statistics panel
        self.ax_stats = self.fig.add_subplot(self.gs[2])
        self.ax_stats.axis('off')
        
        # Button panel
        self.ax_button = self.fig.add_subplot(self.gs[3])
        self.ax_button.axis('off')
        
        # Create input widgets
        self.create_input_widgets()
        
        # Initial message
        self.show_welcome_message()
        
    def create_input_widgets(self):
        """Create ticker input and buttons"""
        # Ticker input box - adjusted positions to prevent overlap
        ax_textbox = plt.axes([0.15, 0.915, 0.15, 0.035])
        self.textbox = TextBox(ax_textbox, 'Ticker Symbol:', initial='AAPL', 
                               color='white', hovercolor='#ecf0f1')
        self.textbox.label.set_fontsize(11)
        self.textbox.label.set_fontweight('bold')
        
        # Load Data button - better spacing
        ax_load = plt.axes([0.32, 0.915, 0.11, 0.035])
        self.btn_load = Button(ax_load, 'Load Data', color='#3498db', hovercolor='#2980b9')
        self.btn_load.label.set_color('white')
        self.btn_load.label.set_fontsize(10)
        self.btn_load.label.set_fontweight('bold')
        self.btn_load.on_clicked(self.load_stock_data)
        
        # Run Simulation button - better spacing
        ax_simulate = plt.axes([0.45, 0.915, 0.14, 0.035])
        self.btn_simulate = Button(ax_simulate, 'Run Simulation', color='#95a5a6', hovercolor='#7f8c8d')
        self.btn_simulate.label.set_color('white')
        self.btn_simulate.label.set_fontsize(10)
        self.btn_simulate.label.set_fontweight('bold')
        self.btn_simulate.on_clicked(self.start_simulation)
        
        # Info text - adjusted position
        info_text = "Enter a ticker symbol and click 'Load Data' to fetch historical data and calculate volatility"
        self.ax_input.text(0.5, 0.2, info_text, transform=self.ax_input.transAxes,
                          fontsize=10, ha='center', va='center', color='#7f8c8d',
                          style='italic')
    
    def show_welcome_message(self):
        """Display welcome message on chart"""
        self.ax_chart.clear()
        self.ax_chart.set_facecolor('#fafafa')
        self.ax_chart.text(0.5, 0.5, 'Enter a ticker symbol above and click "Load Data" to begin',
                          transform=self.ax_chart.transAxes, fontsize=16,
                          ha='center', va='center', color='#95a5a6',
                          bbox=dict(boxstyle='round,pad=1', facecolor='white', 
                                   edgecolor='#bdc3c7', linewidth=2))
        self.ax_chart.set_xticks([])
        self.ax_chart.set_yticks([])
        self.fig.canvas.draw_idle()
    
    def load_stock_data(self, event):
        """Fetch stock data from yfinance and calculate parameters"""
        ticker_symbol = self.textbox.text.strip().upper()
        
        if not ticker_symbol:
            self.show_error("Please enter a ticker symbol")
            return
        
        # Update button state
        self.btn_load.label.set_text('Loading...')
        self.btn_load.ax.set_facecolor('#95a5a6')
        self.fig.canvas.draw_idle()
        
        try:
            # Fetch stock data
            self.ticker = yf.Ticker(ticker_symbol)
            
            # Get 1 year of historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            hist_data = self.ticker.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                raise ValueError(f"No data found for ticker {ticker_symbol}")
            
            # Get current price
            self.current_price = hist_data['Close'].iloc[-1]
            
            # Calculate volatility (annualized)
            returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            self.volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate trading days for 1 year into the future (252 trading days per year)
            self.days_remaining = 252
            
            # Calculate the actual end date (1 year from now)
            self.forecast_end_date = datetime.now() + timedelta(days=365)
            
            # Get stock info
            try:
                info = self.ticker.info
                self.company_name = info.get('longName', ticker_symbol)
                self.exchange = info.get('exchange', 'N/A')
            except:
                self.company_name = ticker_symbol
                self.exchange = 'N/A'
            
            # Set simulation parameters
            self.num_simulations = 100
            self.dt = 1/252
            
            # Define drift scenarios based on historical performance
            hist_return = (self.current_price / hist_data['Close'].iloc[0]) - 1
            self.bull_drift = max(0.20, hist_return * 1.5)  # Optimistic
            self.moderate_drift = hist_return  # Continue historical trend
            self.bear_drift = min(-0.20, hist_return * 0.3)  # Pessimistic
            
            # Store data
            self.stock_data = hist_data
            
            # Generate simulations
            self.generate_simulations()
            
            # Display loaded data
            self.display_stock_info()
            
            # Enable simulation button
            self.btn_simulate.ax.set_facecolor('#27ae60')
            self.btn_simulate.set_active(True)
            self.simulation_ready = True
            
            # Reset load button
            self.btn_load.label.set_text('Load Data')
            self.btn_load.ax.set_facecolor('#3498db')
            
        except Exception as e:
            self.show_error(f"Error loading {ticker_symbol}: {str(e)}")
            self.btn_load.label.set_text('Load Data')
            self.btn_load.ax.set_facecolor('#3498db')
            self.simulation_ready = False
        
        self.fig.canvas.draw_idle()
    
    def generate_simulations(self):
        """Generate Monte Carlo simulation paths for all scenarios"""
        np.random.seed(42)
        
        # Initialize arrays
        self.bull_paths = np.zeros((self.num_simulations, self.days_remaining + 1))
        self.moderate_paths = np.zeros((self.num_simulations, self.days_remaining + 1))
        self.bear_paths = np.zeros((self.num_simulations, self.days_remaining + 1))
        
        # Set initial price
        self.bull_paths[:, 0] = self.current_price
        self.moderate_paths[:, 0] = self.current_price
        self.bear_paths[:, 0] = self.current_price
        
        # Generate paths using geometric Brownian motion
        for i in range(self.num_simulations):
            # Bull scenario
            for t in range(1, self.days_remaining + 1):
                drift_term = (self.bull_drift - 0.5 * self.volatility**2) * self.dt
                shock_term = self.volatility * np.sqrt(self.dt) * np.random.normal()
                self.bull_paths[i, t] = self.bull_paths[i, t-1] * np.exp(drift_term + shock_term)
            
            # Moderate scenario
            for t in range(1, self.days_remaining + 1):
                drift_term = (self.moderate_drift - 0.5 * self.volatility**2) * self.dt
                shock_term = self.volatility * np.sqrt(self.dt) * np.random.normal()
                self.moderate_paths[i, t] = self.moderate_paths[i, t-1] * np.exp(drift_term + shock_term)
            
            # Bear scenario
            for t in range(1, self.days_remaining + 1):
                drift_term = (self.bear_drift - 0.5 * self.volatility**2) * self.dt
                shock_term = self.volatility * np.sqrt(self.dt) * np.random.normal()
                self.bear_paths[i, t] = self.bear_paths[i, t-1] * np.exp(drift_term + shock_term)
        
        self.time_points = np.arange(0, self.days_remaining + 1)
    
    def display_stock_info(self):
        """Display loaded stock information"""
        self.ax_chart.clear()
        self.ax_chart.set_facecolor('#fafafa')
        
        info_text = f"{self.company_name} ({self.textbox.text.upper()})\n\n"
        info_text += f"Current Price: ${self.current_price:.2f}\n"
        info_text += f"Historical Volatility: {self.volatility*100:.2f}% (annualized)\n"
        info_text += f"Exchange: {self.exchange}\n\n"
        info_text += f"📊 Drift Scenarios:\n"
        info_text += f"  • Bull: {self.bull_drift*100:+.2f}% annual\n"
        info_text += f"  • Moderate: {self.moderate_drift*100:+.2f}% annual\n"
        info_text += f"  • Bear: {self.bear_drift*100:+.2f}% annual\n\n"
        info_text += f"Ready to simulate {self.num_simulations} paths for {self.days_remaining} trading days"
        
        self.ax_chart.text(0.5, 0.5, info_text, 
                          transform=self.ax_chart.transAxes, fontsize=13,
                          ha='center', va='center', color='#2c3e50',
                          bbox=dict(boxstyle='round,pad=1.5', facecolor='white', 
                                   edgecolor='#3498db', linewidth=3))
        self.ax_chart.set_xticks([])
        self.ax_chart.set_yticks([])
        self.fig.canvas.draw_idle()
    
    def show_error(self, message):
        """Display error message"""
        self.ax_chart.clear()
        self.ax_chart.set_facecolor('#fafafa')
        self.ax_chart.text(0.5, 0.5, f'⚠ {message}',
                          transform=self.ax_chart.transAxes, fontsize=14,
                          ha='center', va='center', color='#e74c3c',
                          bbox=dict(boxstyle='round,pad=1', facecolor='#fff5f5', 
                                   edgecolor='#e74c3c', linewidth=2))
        self.ax_chart.set_xticks([])
        self.ax_chart.set_yticks([])
        self.fig.canvas.draw_idle()
    
    def start_simulation(self, event):
        """Start the animated Monte Carlo simulation"""
        if not self.simulation_ready:
            self.show_error("Please load stock data first")
            return
        
        # Disable simulation button during animation
        self.btn_simulate.label.set_text('Simulating...')
        self.btn_simulate.ax.set_facecolor('#95a5a6')
        self.btn_simulate.set_active(False)
        
        # Clear the statistics panel at the start of simulation
        self.clear_statistics()
        
        # Reset drawn paths
        self.drawn_paths = {'bull': [], 'moderate': [], 'bear': []}
        self.current_path = 0
        
        # Create animation
        self.anim = FuncAnimation(self.fig, self.animate_simulation, 
                                 frames=self.num_simulations, interval=50,
                                 repeat=False, blit=False)
        
        self.fig.canvas.draw_idle()
    
    def clear_statistics(self):
        """Clear the statistics panel"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        self.fig.canvas.draw_idle()
    
    def animate_simulation(self, frame):
        """Animation function for each frame"""
        # Store paths for current frame
        self.drawn_paths['bull'].append(self.bull_paths[frame, :])
        self.drawn_paths['moderate'].append(self.moderate_paths[frame, :])
        self.drawn_paths['bear'].append(self.bear_paths[frame, :])
        
        self.current_path = frame + 1
        
        # Clear and redraw chart
        self.ax_chart.clear()
        self.ax_chart.set_facecolor('#fafafa')
        
        # Calculate alpha based on number of paths
        alpha = max(0.1, min(0.4, 10.0 / self.current_path))
        
        # Plot all paths with color coding
        # Bull case (green)
        for path in self.drawn_paths['bull']:
            self.ax_chart.plot(self.time_points, path, color='#27ae60', 
                              alpha=alpha, linewidth=0.8)
        
        # Moderate case (yellow/orange)
        for path in self.drawn_paths['moderate']:
            self.ax_chart.plot(self.time_points, path, color='#f39c12', 
                              alpha=alpha, linewidth=0.8)
        
        # Bear case (red)
        for path in self.drawn_paths['bear']:
            self.ax_chart.plot(self.time_points, path, color='#e74c3c', 
                              alpha=alpha, linewidth=0.8)
        
        # Plot median paths if we have data
        if self.current_path > 0:
            bull_array = np.array(self.drawn_paths['bull'])
            moderate_array = np.array(self.drawn_paths['moderate'])
            bear_array = np.array(self.drawn_paths['bear'])
            
            self.ax_chart.plot(self.time_points, np.median(bull_array, axis=0),
                              color='#27ae60', linewidth=3, 
                              label=f'Bull Case ({self.bull_drift*100:.1f}%)', 
                              linestyle='-', zorder=10)
            self.ax_chart.plot(self.time_points, np.median(moderate_array, axis=0),
                              color='#f39c12', linewidth=3, 
                              label=f'Moderate Case ({self.moderate_drift*100:.1f}%)', 
                              linestyle='-', zorder=10)
            self.ax_chart.plot(self.time_points, np.median(bear_array, axis=0),
                              color='#e74c3c', linewidth=3, 
                              label=f'Bear Case ({self.bear_drift*100:.1f}%)', 
                              linestyle='-', zorder=10)
        
        # Current price line
        self.ax_chart.axhline(y=self.current_price, color='#34495e', 
                             linestyle='--', linewidth=2, 
                             label=f'Current: ${self.current_price:.2f}', zorder=5)
        
        # Formatting
        self.ax_chart.set_xlabel('Trading Days (1 Year Forecast)', 
                                fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax_chart.set_ylabel('Stock Price ($)', 
                                fontsize=12, fontweight='bold', color='#2c3e50')
        
        title = f'{self.company_name} ({self.textbox.text.upper()}) | '
        title += f'Current: ${self.current_price:.2f} | '
        title += f'Volatility: {self.volatility*100:.1f}% | '
        title += f'Paths: {self.current_path}/{self.num_simulations}'
        
        self.ax_chart.set_title(title, fontsize=13, color='#2c3e50', 
                               pad=15, fontweight='bold')
        self.ax_chart.legend(loc='best', framealpha=0.95, fontsize=10)
        self.ax_chart.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax_chart.set_xlim(0, self.days_remaining)
        
        # Dynamic y-axis
        if self.current_path > 0:
            all_paths = np.concatenate([bull_array, moderate_array, bear_array])
            y_min = max(0, all_paths.min() * 0.9)
            y_max = all_paths.max() * 1.1
            self.ax_chart.set_ylim(y_min, y_max)
        
        # Update statistics when simulation is complete
        if self.current_path >= self.num_simulations:
            self.display_statistics()
            
            # Reset button
            self.btn_simulate.label.set_text('Run Simulation')
            self.btn_simulate.ax.set_facecolor('#27ae60')
            self.btn_simulate.set_active(True)
            
            if self.anim is not None:
                self.anim.event_source.stop()
    
    def calculate_statistics(self):
        """Calculate statistics from the simulations"""
        bull_final = self.bull_paths[:, -1]
        moderate_final = self.moderate_paths[:, -1]
        bear_final = self.bear_paths[:, -1]
        
        stats = {
            'bull': {
                'median': np.median(bull_final),
                'mean': np.mean(bull_final),
                'p10': np.percentile(bull_final, 10),
                'p90': np.percentile(bull_final, 90),
                'median_return': ((np.median(bull_final) / self.current_price - 1) * 100),
                'mean_return': ((np.mean(bull_final) / self.current_price - 1) * 100)
            },
            'moderate': {
                'median': np.median(moderate_final),
                'mean': np.mean(moderate_final),
                'p10': np.percentile(moderate_final, 10),
                'p90': np.percentile(moderate_final, 90),
                'median_return': ((np.median(moderate_final) / self.current_price - 1) * 100),
                'mean_return': ((np.mean(moderate_final) / self.current_price - 1) * 100)
            },
            'bear': {
                'median': np.median(bear_final),
                'mean': np.mean(bear_final),
                'p10': np.percentile(bear_final, 10),
                'p90': np.percentile(bear_final, 90),
                'median_return': ((np.median(bear_final) / self.current_price - 1) * 100),
                'mean_return': ((np.mean(bear_final) / self.current_price - 1) * 100)
            }
        }
        
        return stats
    
    def display_statistics(self):
        """Display simulation statistics below the chart"""
        stats = self.calculate_statistics()
        
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Create formatted statistics table
        stats_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          ONE-YEAR FORECAST RESULTS ({self.forecast_end_date.strftime('%B %d, %Y')})                            ║
║                                     Based on {self.num_simulations} Monte Carlo Simulations                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  BULL SCENARIO       │  Mean: ${stats['bull']['mean']:>8.2f} ({stats['bull']['mean_return']:>+6.2f}%)  │  Median: ${stats['bull']['median']:>8.2f} ({stats['bull']['median_return']:>+6.2f}%)  │  90%: ${stats['bull']['p90']:>8.2f}  ║
║  MODERATE SCENARIO   │  Mean: ${stats['moderate']['mean']:>8.2f} ({stats['moderate']['mean_return']:>+6.2f}%)  │  Median: ${stats['moderate']['median']:>8.2f} ({stats['moderate']['median_return']:>+6.2f}%)  │  90%: ${stats['moderate']['p90']:>8.2f}  ║
║  BEAR SCENARIO       │  Mean: ${stats['bear']['mean']:>8.2f} ({stats['bear']['mean_return']:>+6.2f}%)  │  Median: ${stats['bear']['median']:>8.2f} ({stats['bear']['median_return']:>+6.2f}%)  │  10%: ${stats['bear']['p10']:>8.2f}  ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Current Price: ${self.current_price:>8.2f}  │  Trading Days: {self.days_remaining:>3}  │  Historical Volatility: {self.volatility*100:>5.2f}%                               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        self.ax_stats.text(0.5, 0.5, stats_text.strip(), 
                          transform=self.ax_stats.transAxes,
                          fontsize=8.5, verticalalignment='center', 
                          horizontalalignment='center',
                          fontfamily='monospace', color='#2c3e50',
                          bbox=dict(boxstyle='round,pad=1', facecolor='#ffffff', 
                                   edgecolor='#bdc3c7', linewidth=2))
        
        # Force redraw to ensure statistics appear
        self.fig.canvas.draw_idle()
    
    def run(self):
        """Start the GUI"""
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point"""
    # Check for required packages
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas")
        return
    
    try:
        import yfinance as yf
    except ImportError:
        print("Error: yfinance is required. Install with: pip install yfinance")
        return
    
    print("=" * 60)
    print("  Monte Carlo Stock Price Simulator")
    print("  Using Real Market Data from Yahoo Finance")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Enter a ticker symbol (e.g., AAPL, TSLA, MSFT)")
    print("2. Click 'Load Data' to fetch historical data")
    print("3. Click 'Run Simulation' to generate Monte Carlo paths")
    print("\nThe simulation will forecast prices 1 year into the future")
    print("based on historical volatility and multiple scenarios.")
    print("=" * 60)
    print()
    
    gui = StockMonteCarloGUI()
    gui.run()


if __name__ == "__main__":
    main()
