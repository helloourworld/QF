import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# Download historical data for USD/CAD
symbol = "CAD=X"  # Yahoo Finance symbol for USD/CAD
usd_cad = yf.download(symbol, start="2023-01-01")

# Calculate Stochastic Oscillator
low_14 = usd_cad['Low'].rolling(window=14).min()
high_14 = usd_cad['High'].rolling(window=14).max()

usd_cad['%K'] = ((usd_cad['Close'] - low_14) / (high_14 - low_14)) * 100
usd_cad['%D'] = usd_cad['%K'].rolling(window=3).mean()

# Drop rows with NaN values
usd_cad.dropna(inplace=True)

# Plotting the Stochastic Oscillator
fig = go.Figure()
fig.add_trace(go.Scatter(x=usd_cad.index, y=usd_cad['%K'], mode='lines', name='%K'))
fig.add_trace(go.Scatter(x=usd_cad.index, y=usd_cad['%D'], mode='lines', name='%D'))
fig.add_shape(type="line", x0=usd_cad.index[0], x1=usd_cad.index[-1], y0=80, y1=80,
              line=dict(color="red", dash="dash"), name="Overbought")
fig.add_shape(type="line", x0=usd_cad.index[0], x1=usd_cad.index[-1], y0=20, y1=20,
              line=dict(color="green", dash="dash"), name="Oversold")

fig.update_layout(title="Stochastic Oscillator for USD/CAD",
                  xaxis_title="Date",
                  yaxis_title="Oscillator Value",
                  legend_title="Lines")

# # Save the plot
# fig.write_json("usd_cad_stochastic.json")
# fig.write_image("usd_cad_stochastic.png")
fig.show()