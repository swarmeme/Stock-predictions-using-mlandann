
# Stock Price Prediction using LSTM

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)

---

Predicting stock prices using Long Short-Term Memory (LSTM) neural networks based on historical stock price data. In this project, we'll demonstrate how to build a simple LSTM model to predict future stock prices for Apple Inc. (AAPL). This can serve as a starting point for more complex financial prediction models.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Building the LSTM Model](#building-the-lstm-model)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Evaluation](#evaluation)
- [Predicting Future Prices](#predicting-future-prices)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- TensorFlow 2.0+ (for deep learning)
- yfinance (for downloading stock data)
- pandas, numpy, scikit-learn, matplotlib (for data manipulation and visualization)

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/swarmeme/Stock-predictions-using-mlandann.git
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter Notebook `stock_price_prediction.ipynb` to run the code step by step.

2. Follow the comments and explanations in the notebook to understand each part of the code.

3. Experiment with the model architecture, hyperparameters, and dataset to improve predictions.

## Data Preparation

- Historical stock price data for Apple Inc. is downloaded using the `yfinance` library.
- Data is preprocessed and scaled using Min-Max scaling to fit the LSTM model.

## Building the LSTM Model

- A sequential LSTM model is defined with two LSTM layers, each containing 50 neurons, followed by two Dense layers.
- The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss.

## Training the Model

- The model is trained on a portion (80%) of the data using a batch size of 1 and one epoch.

## Making Predictions

- The model makes predictions on the remaining data, and the predictions are inverse scaled to obtain actual stock prices.

## Evaluation

- The Root Mean Squared Error (RMSE) is calculated to evaluate the accuracy of the model's predictions.

## Predicting Future Prices

- The model is used to predict the next day's closing price based on the last 60 days of closing prices.

---

**Disclaimer**: Predicting stock prices is a complex task, and this project serves as a basic example. Actual stock market behavior is influenced by various factors, and this model may not provide accurate predictions for real-world trading decisions. Use it for educational purposes and consider seeking financial advice from professionals before making investment decisions.
