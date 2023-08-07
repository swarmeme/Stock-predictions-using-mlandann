import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #to scale the data
from keras.models import Sequential #one tensor ip to one tensor op
from keras.layers import Dense, LSTM #dense is all the activation func biases etc
import matplotlib.pyplot as plt
import yfinance as yf
plt.style.use("fivethirtyeight")

# %%
#stock quote
start = '2012-01-01'
end = '2020-12-31'
dataFrame = yf.download('AAPL', start=start, end=end) #stock info from 2012-2020

#get the number of rows and columns
dataFrame.shape

# %%
#visualise the closing price history
plt.figure(figsize=(16,8))
plt.title("Closing Price")
plt.plot(dataFrame['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closing Price USD ($)', fontsize=18)
plt.show()