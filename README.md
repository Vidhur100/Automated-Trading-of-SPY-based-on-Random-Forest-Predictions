# Automated-Trading-of-SPY-based-on-Random-Forest-Predictions
This project uses machine learning models to algorithmically trade the SPY ETF, which tracks the S&P 500 index.
# Overview
- Uses a Random Forest classifier to predict if SPY will increase or decrease each day
- Trains model on historical data from 2010-2022
- Makes predictions on 2023 data and simulates trades
- Calculates profitability of simulated trades
# Demo
https://github.com/Vidhur100/Automated-Trading-of-SPY-based-on-Random-Forest-Predictions/assets/11946297/1a5a216f-0a2e-4d34-9125-ef3b47b47826
# Usage
The main file to run is trading_bot.py. Simply run:
python trading_bot.py
This will:
- Download historical SPY data
- Train model
- Make predictions and simulate trades
- Plot chart with signals
- Print trade details and profitability
# Customization
The following parts can be customized:
- Date range for training data
- Model parameters (tune RandomForest hyperparams)
- Trailing stop loss for trade exit
- Position sizing/risk management
# Requirements
The required Python packages are:
yfinance
pandas
sklearn
matplotlib
Install via pip install -r requirements.txt
# Contributing
Pull requests and feature additions are welcome!
Some ideas for improvements:
Additional models for prediction
Optimization of trades
Backtesting over multiple time periods
# Code Steps
1. Import necessary python libraries for data analysis and modeling
2. Use yfinance to download historical price data for SPY (S&P 500 ETF) from 2010 to 2023
3. Extract the Open, High, Low, Close, and Volume columns as model features (X)
4. Create a target variable (y) that is 1 if Close > Open (price went up) and 0 otherwise
5. Split the data into training and test sets to evaluate model performance
6. Train a Random Forest classifier model on the training data to predict if price will go up or down
7. Evaluate model accuracy on the test set to see how well it predicts correctly
8. Get SPY price data for 2023 and make predictions with the model
9. Plot the 2023 price chart and add green ^ markers for predicted price increases and red v markers for decreases
10. Simulate trades by going long when the model predicts an increase and short when it predicts a decrease
11. Calculate profit/loss for each simulated trade based on entry and exit price
12. Print out a table summarizing each simulated trade
13. Calculate total profit from all simulated trades
