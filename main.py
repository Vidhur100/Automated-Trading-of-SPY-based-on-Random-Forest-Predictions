import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import Random Forest model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download SPY data from 2010-2023
spy = yf.Ticker("SPY")
hist = spy.history(start="2010-01-01", end="2023-01-01")

# Extract features (Open, High, Low, Close, Volume) and target (Close > Open?)
X = hist[["Open", "High", "Low", "Close", "Volume"]]
y = (hist["Close"] > hist["Open"]).astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classification model on training data
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model performance on test set
predictions = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))

# Get SPY data for 2023
data_2023 = spy.history(start="2023-01-01", end="2023-07-15")

# Generate predictions on 2023 data
predictions_2023 = model.predict(data_2023[["Open", "High", "Low", "Close", "Volume"]])

# Plot price chart with buy/sell signals based on predictions
plt.plot(data_2023.index, data_2023["Close"])
for i in range(len(predictions_2023)):
    if predictions_2023[i] == 1: # Predict price increase
        plt.scatter(data_2023.index[i], data_2023["Close"][i], c="green", marker="^", s=100)
    else: # Predict price decrease
        plt.scatter(data_2023.index[i], data_2023["Close"][i], c="red", marker="v", s=100)
plt.title("SPY Price Chart with Entry/Exit Signals")
plt.show()

# Print details of simulated trades
trades = []
current_position = None
for i in range(len(predictions_2023)):
    if predictions_2023[i] == 1 and current_position == None: # Enter long trade
        buy_price = data_2023["Close"][i]
        current_position = "Long"
    elif predictions_2023[i] == 0 and current_position == "Long": # Exit long trade
        sell_price = data_2023["Close"][i]
        pl_pct = (sell_price / buy_price - 1) * 100
        trades.append({"Buy Price": buy_price, "Sell Price": sell_price, "P/L %": pl_pct})
        current_position = None
    elif predictions_2023[i] == 0 and current_position == None: # Enter short trade
        sell_price = data_2023["Close"][i]
        current_position = "Short"
    elif predictions_2023[i] == 1 and current_position == "Short": # Exit short trade
        buy_price = data_2023["Close"][i]
        pl_pct = (sell_price / buy_price - 1) * 100
        trades.append({"Buy Price": buy_price, "Sell Price": sell_price, "P/L %": pl_pct})
        current_position = None

print(pd.DataFrame(trades))

# Calculate total profit from simulated trades
total_profit = sum([trade["P/L %"] for trade in trades])
print("Total Profit: {:.2f}%".format(total_profit))
