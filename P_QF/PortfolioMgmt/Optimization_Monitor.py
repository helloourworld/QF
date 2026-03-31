import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DEFINE TICKERS TO MONITOR
# We include AVGO peers and AI/Semiconductor ETFs
tickers = {
    "AVGO": "Broadcom (Focus)",
    "NVDA": "NVIDIA (Peer)",
    "AMD": "AMD (Peer)",
    "MRVL": "Marvell (Custom Chip Peer)",
    "MSFT": "Microsoft (Software Client)",
    "GOOGL": "Google (Software Client)",
    "SOXX": "iShares Semi ETF",
    "SMH": "VanEck Semi ETF"
}

def get_valuation_metrics(ticker_dict):
    rows = []
    print("Fetching real-time 2026 valuation data...")
    
    for symbol, label in ticker_dict.items():
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Calculate FCF Yield (Free Cash Flow / Market Cap)
        fcf = info.get('freeCashflow', 0)
        mcap = info.get('marketCap', 1) # avoid div by zero
        fcf_yield = (fcf / mcap) * 100 if fcf else 0
        # print(info.keys())
        data = {
            "Symbol": symbol,
            "Type": label,
            "Price": info.get('currentPrice'),
            "Forward P/E": info.get('forwardPE'),
            "Trailing P/E": info.get('trailingPE'),
            "PEG Ratio": info.get('trailingPegRatio'), # < 1.0 is considered undervalued for growth
            "EV/EBITDA": info.get('enterpriseToEbitda'),
            "FCF Yield %": round(fcf_yield, 2),
            "Div Yield %": (info.get('dividendYield', 0) or 0) * 100,
            "P/S Ratio": info.get('priceToSalesTrailing12Months')
        }
        rows.append(data)
        
    return pd.DataFrame(rows)

# 2. FETCH AND CLEAN DATA
df = get_valuation_metrics(tickers)
df.set_index("Symbol", inplace=True)

# 3. GENERATE BIRD'S EYE VIEW SUMMARY
print("\n--- VALUATION SUMMARY TABLE ---")
print(df[["Type", "Forward P/E", "PEG Ratio", "FCF Yield %", "Div Yield %"]].sort_values(by="Forward P/E"))

# 4. VISUALIZATION
# Add value labels on top of bars
def add_value_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if height != 0:  # Only label non-zero values
            ax.text(p.get_x() + p.get_width()/2., height,
                   f'{height:.2f}',
                   ha="center", va="bottom", fontsize=9)
            
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Create a 2x2 subplot
# Subplot 1: Forward P/E Comparison
plt.subplot(2, 2, 1)
sns.barplot(x=df.index, y=df['Forward P/E'], hue=df.index, legend=False)
add_value_labels(plt.gca())
plt.title("Forward P/E Ratio (Lower is Cheaper)")
plt.axhline(35, ls='--', color='red', label='High P/E Threshold')

# Subplot 2: PEG Ratio (Growth Adjusted Valuation)
plt.subplot(2, 2, 2)
sns.barplot(x=df.index, y=df['PEG Ratio'], hue=df.index, legend=False)
add_value_labels(plt.gca())
plt.title("PEG Ratio (Below 1.0 is Good Value)")
plt.axhline(1.0, ls='--', color='blue')

# Subplot 3: Free Cash Flow Yield
plt.subplot(2, 2, 3)
sns.barplot(x=df.index, y=df['FCF Yield %'], hue=df.index, legend=False)
add_value_labels(plt.gca())
plt.title("Free Cash Flow Yield % (Higher is Better)")

# Subplot 4: Dividend Yield
plt.subplot(2, 2, 4)
sns.barplot(x=df.index, y=df['Div Yield %'], hue=df.index, legend=False)
add_value_labels(plt.gca())
plt.title("Dividend Yield %")

plt.tight_layout()
plt.show()