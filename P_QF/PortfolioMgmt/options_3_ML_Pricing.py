# -*- coding: utf-8 -*-
"""
@author: lyu
1 Data Generation
2 Option Pricing
3 Volatility Skew Adjustment
4 Data Splitting
5 Machine Learning Models
6 Summary of Model Performance
7 Scatter Plots of Actual vs. Predicted Option Prices

"""

# 1 Data Generation
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import numpy as np
import pandas as pd
# Set random seed for reproducibility
np.random.seed(42)
# Number of samples
n_samples = 100000


# Generating random parameters for the options
S = np.random.uniform(50, 150, n_samples)  # Spot Price: Between $50 and $150
K = np.random.uniform(50, 150, n_samples)  # Strike Price: Between $50 and $150
# Time to Maturity: Between 3 months␣
T = np.random.uniform(0.25, 2, n_samples)
# and 2 years
r = np.random.uniform(0.01, 0.05, n_samples)  # Risk-free Rate: Between 1% and␣
# 5%
# Introducing volatility skew: Options with lower strike prices will tend to␣
# have higher volatilities
sigma = np.random.uniform(0.1, 0.4, n_samples) + \
    (K < S) * np.random.uniform(0.05, 0.15, n_samples)
# Creating a DataFrame to store these values
options_df = pd.DataFrame({
    'Spot_Price': S,
    'Strike_Price': K,
    'Time_to_Maturity': T,
    'Risk_free_Rate': r,
    'Volatility': sigma
})
options_df.head()

# 2 Option Pricing


def black_scholes_put_price(S, K, T, r, sigma, q=0):
    """
    Compute the Black-Scholes put option price.
    Parameters:
    - S: Spot price of the underlying asset
    - K: Strike price of the option
    - T: Time to maturity (in years)
    - r: Risk-free interest rate (annualized)
    - sigma: Volatility of the underlying asset (annualized)
    - **q: Dividend yield (annualized). Default is 0 (no dividends).**
    Returns:
    - Put option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - \
        S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


# Compute put option prices using Black-Scholes formula
options_df['BS_Put_Price'] = black_scholes_put_price(
    options_df['Spot_Price'],
    options_df['Strike_Price'],
    options_df['Time_to_Maturity'],
    options_df['Risk_free_Rate'],
    options_df['Volatility']
)
options_df[['Spot_Price', 'Strike_Price', 'Volatility', 'BS_Put_Price']].head()


# 3 Volatility Skew Adjustment

# The volatility skew is a phenomenon where out-of-the-money (OTM) options tend to have higher
# implied volatilities compared to at-the-money (ATM) or in-the-money (ITM) options. This skew
# is often more pronounced for options with lower strike prices.
"""
The volatility skew is affected by sentiment and the supply and demand relationship of particular
 options in the market. It provides information on whether traders and investors prefer to write 
 calls or puts. Traders can use relative changes in skew for an options series as a trading 
 strategy. 

"""

# Adjusting the volatility skew based on how far OTM the option is and the␣
# magnitude of the strike price
delta_from_ATM = (options_df['Spot_Price'] - options_df['Strike_Price']).abs()
skew_factor = np.where(options_df['Strike_Price'] < options_df['Spot_Price'],
                       delta_from_ATM / options_df['Strike_Price'], 0)

# Adjusting the volatility to introduce a more pronounced skew for OTM options and options with smaller strike prices
options_df['Adjusted_Volatility'] = options_df['Volatility'] + \
    skew_factor * np.random.uniform(0.05, 0.2, n_samples)
# Compute put option prices using Black-Scholes formula with adjusted volatility
options_df['Adjusted_BS_Put_Price'] = black_scholes_put_price(
    options_df['Spot_Price'],
    options_df['Strike_Price'],
    options_df['Time_to_Maturity'],
    options_df['Risk_free_Rate'],
    options_df['Adjusted_Volatility']
)
options_df[['Spot_Price', 'Strike_Price', 'Volatility',
            'BS_Put_Price', 'Adjusted_BS_Put_Price']].head()


# 4 Data Splitting
# We’ll split the generated data into training and testing sets. This ensures that the machine learning
# models are evaluated on unseen data, providing a more realistic assessment of their predictive
# capabilities.
# Features and target variable
X = options_df[['Spot_Price', 'Strike_Price',
                'Time_to_Maturity', 'Risk_free_Rate', 'Volatility']]
y = options_df['Adjusted_BS_Put_Price']
# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
X_train.shape, X_test.shape


# 5 Machine Learning Models
# 5.1 Linear Regression Model


# Initialize the Linear Regression model
lr_model = LinearRegression()
# Train the model
lr_model.fit(X_train, y_train)
# Predict on the test set
lr_predictions = lr_model.predict(X_test)
# Compute accuracy metrics
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
lr_mse, lr_mae, lr_r2

# 5.2 Decision Tree Regressor
# Initialize the Decision Tree Regressor model
dt_model = DecisionTreeRegressor(random_state=42)
# Train the model
dt_model.fit(X_train, y_train)
# Predict on the test set
dt_predictions = dt_model.predict(X_test)
# Compute accuracy metrics
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)
dt_mse, dt_mae, dt_r2

# 5.3 Random Forest Regressor
# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model
rf_model.fit(X_train, y_train)
# Predict on the test set
rf_predictions = rf_model.predict(X_test)
# Compute accuracy metrics
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
rf_mse, rf_mae, rf_r2


# 5.4 Gradient Boosted Trees Regressor
# Gradient Boosting is a powerful
# ensemble technique that can provide high accuracy by iteratively correcting errors from previous
# trees. Let’s train, predict, and evaluate this model.

# Initialize the Gradient Boosting Regressor model

gb_model = GradientBoostingRegressor(random_state=42)
# Train the model
gb_model.fit(X_train, y_train)
# Predict on the test set
gb_predictions = gb_model.predict(X_test)
# Compute accuracy metrics
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)
gb_mse, gb_mae, gb_r2


# 5.5 Neural Network (MLP) Regressor
# Lastly, we’ll employ a Neural Network model. Neural Networks can capture complex non-linear
# relationships in data and are particularly suited for large datasets. Before training the model, we’ll
# scale our features since neural networks perform better with standardized input data.

# Neural networks benefit from feature scaling. So, we'll scale our features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize the Neural Network (MLP) model
nn_model = MLPRegressor(hidden_layer_sizes=(
    100, 50), max_iter=1000, random_state=42)
# Train the model
nn_model.fit(X_train_scaled, y_train)
# Predict on the test set
nn_predictions = nn_model.predict(X_test_scaled)
# Compute accuracy metrics
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)
nn_mse, nn_mae, nn_r2


# 6 Summary of Model Performance

# Constructing the results dataframe
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosted Trees', 'Neural Network (MLP)'],
    'Mean Squared Error': [lr_mse, dt_mse, rf_mse, gb_mse, nn_mse],
    'Mean Absolute Error': [lr_mae, dt_mae, rf_mae, gb_mae, nn_mae],
    'R-squared': [lr_r2, dt_r2, rf_r2, gb_r2, nn_r2]
})
results_df

# 7 Scatter Plots of Actual vs. Predicted Option Prices
# Set up the figure and axes
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('Scatter Plots of Actual vs. Predicted Option Prices', fontsize=16)
# Linear Regression Scatter Plot
axs[0, 0].scatter(y_test, lr_predictions, alpha=0.5)
axs[0, 0].plot([min(y_test), max(y_test)], [
               min(y_test), max(y_test)], '--', color='red')
axs[0, 0].set_title('Linear Regression')
axs[0, 0].set_xlabel('Actual Prices')
axs[0, 0].set_ylabel('Predicted Prices')
# Decision Tree Scatter Plot
axs[0, 1].scatter(y_test, dt_predictions, alpha=0.5)
axs[0, 1].plot([min(y_test), max(y_test)], [
               min(y_test), max(y_test)], '--', color='red')
axs[0, 1].set_title('Decision Tree')
axs[0, 1].set_xlabel('Actual Prices')
axs[0, 1].set_ylabel('Predicted Prices')
# Random Forest Scatter Plot
axs[0, 2].scatter(y_test, rf_predictions, alpha=0.5)
axs[0, 2].plot([min(y_test), max(y_test)], [
               min(y_test), max(y_test)], '--', color='red')
axs[0, 2].set_title('Random Forest')
axs[0, 2].set_xlabel('Actual Prices')
axs[0, 2].set_ylabel('Predicted Prices')
# Gradient Boosted Trees Scatter Plot
axs[1, 0].scatter(y_test, gb_predictions, alpha=0.5)
axs[1, 0].plot([min(y_test), max(y_test)], [
               min(y_test), max(y_test)], '--', color='red')
axs[1, 0].set_title('Gradient Boosted Trees')
axs[1, 0].set_xlabel('Actual Prices')
axs[1, 0].set_ylabel('Predicted Prices')
# Neural Network Scatter Plot
axs[1, 1].scatter(y_test, nn_predictions, alpha=0.5)
axs[1, 1].plot([min(y_test), max(y_test)], [
               min(y_test), max(y_test)], '--', color='red')
axs[1, 1].set_title('Neural Network (MLP)')
axs[1, 1].set_xlabel('Actual Prices')
axs[1, 1].set_ylabel('Predicted Prices')
# Hide the empty subplot
axs[1, 2].axis('off')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

"""
Discussion:

The Neural Network’s remarkable
performance highlights the potential of deep learning techniques in the realm of financial modeling.
Its near-perfect predictions underscore its capacity to capture complex nonlinear relationships
inherent in option pricing.

While the Neural Network stood out, tree-based models, particularly the
Random Forest, showcased impressive predictive abilities. These models offer the advantage
of better interpretability, making them invaluable in scenarios where understanding model
decisions is pivotal.

Linear Regression, though foundational, struggled to capture
the complexities of option pricing, evident from its lower R2 value compared to other models.
This suggests that the relationships in the data might be inherently nonlinear, necessitating
more intricate models.
"""
