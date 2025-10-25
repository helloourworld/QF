from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import scipy as sp
import numpy as np
import pandas as pd
import empyrical as ep
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2024-08-31"
df = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Calculating Moving Averages and Bollinger Bands To understand the stock trends better, we’ll calculate the 20-day Simple
# Moving Average (SMA) and the Bollinger Bands. The Bollinger Bands give us an upper and lower range for the stock prices based on the
# 20-day SMA and standard deviation.

df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['Std_Dev'] = df['Close'].rolling(window=20).std()
df['Upper_BB'] = df['SMA_20'] + (2 * df['Std_Dev'])
df['Lower_BB'] = df['SMA_20'] - (2 * df['Std_Dev'])
df = df.dropna()

# Step 3: Training the SVR Model Next, we’ll prepare our features and target variable for the SVR model. We’ll use the SMA and Bollinger
# Bands as features to predict the stock prices.

X = df[['SMA_20', 'Upper_BB', 'Lower_BB']]
y = df['Close']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'kernel': ['rbf', 'linear']
}
grid_search = GridSearchCV(SVR(), param_grid, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_scaled, y)
best_model = grid_search.best_estimator_

# Step 4: Making Predictions Now, with our trained model, let’s predict the future stock prices for the next 30 business days. We’ll also
# visualize the actual prices, predicted prices, and Bollinger Bands.

last_date = df.index[-1]
next_dates = [last_date + timedelta(days=i) for i in range(1, 43)]
next_dates = [date for date in next_dates if date.weekday() < 5][:30]
last_20_prices = df['Close'].tail(20).values
last_20_sma = df['SMA_20'].tail(20).values
last_20_std = df['Std_Dev'].tail(20).values
next_predictions = []
for _ in range(len(next_dates)):
    sma_20 = np.mean(last_20_prices[-20:])
    std_20 = np.std(last_20_prices[-20:])
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    features = np.array([[sma_20, upper_bb, lower_bb]])
    scaled_features = scaler.transform(features)
    prediction = best_model.predict(scaled_features)[0]
    next_predictions.append(prediction)
    last_20_prices = np.append(last_20_prices[1:], prediction)
    last_20_sma = np.append(last_20_sma[1:], sma_20)
    last_20_std = np.append(last_20_std[1:], std_20)
predictions_df = pd.DataFrame({
    'Date': next_dates,
    'Predicted_Price': next_predictions
})

# Step 5: Visualizing the Results Let’s create a beautiful plot to visualize the actual prices, predicted prices, and Bollinger Bands. We’ll use
# Plotly for this…
fig = make_subplots(rows=1, cols=1)
fig.add_trace(
    go.Scatter(x=df.index, y=df['Close'], mode='markers', name='Actual Prices',
               marker=dict(color='blue', size=5))
)
fig.add_trace(
    go.Scatter(x=df.index, y=best_model.predict(X_scaled), mode='lines', name='Predicted Prices',
               line=dict(color='red'))
)
fig.add_trace(
    go.Scatter(x=df.index, y=df['Upper_BB'], mode='lines', name='Upper Bollinger Band',
               line=dict(color='green', dash='dash'))
)
fig.add_trace(
    go.Scatter(x=df.index, y=df['Lower_BB'], mode='lines', name='Lower Bollinger Band',
               line=dict(color='green', dash='dash'))
)
fig.add_trace(
    go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted_Price'],
               mode='markers+lines', name='Future Predictions',
               marker=dict(color='purple', size=8), line=dict(color='purple', dash='dash'))
)
fig.update_layout(
    title=f'{ticker} Stock Price Prediction using SVM with Bollinger Bands',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Legend',
    hovermode="x unified"
)
y_min = min(df['Close'].min(), df['Lower_BB'].min(),
            predictions_df['Predicted_Price'].min())
y_max = max(df['Close'].max(), df['Upper_BB'].max(),
            predictions_df['Predicted_Price'].max())
fig.update_yaxes(range=[y_min * 0.9, y_max * 1.1])
fig.show()

# Step 6: Why Combine Bollinger Bands with SVR? Bollinger Bands provide dynamic upper and lower bounds based on volatility around
# the moving average, offering insights into potential price movements. Support Vector Regression (SVR), on the other hand, excels in
# capturing complex relationships in data and predicting stock prices based on historical trends. Synergy in Prediction By combining
# Bollinger Bands with SVR, we leverage the robustness of SVR in predicting trends while using Bollinger Bands to validate predictions
# within plausible price ranges. This synergy enhances the accuracy and reliability of our stock price forecasts. Visualizing Predictions As
# seen in our plot, the combination allows us to visualize predicted prices alongside Bollinger Bands, offering a comprehensive view of
# potential price movements and helping traders make informed decisions.

"""

"""