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

# %%
#to create dataframe only with close column
data = dataFrame.filter(['Close'])
#convert df into numpy array
dataset = data.values
#number of rows
training_data_len = math.ceil(len(dataset) * .8) #math.ceil rounds upto nearest integer
training_data_len


# %%
#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset) #compute the max n min values for scaling. fit the data into a model and transform it into a form that is more suitable for the model in a single step
scaled_data

# %%
#create the scaled training dataset
train_data = scaled_data[0:training_data_len, :] 
#split the data into x_train and y_train
x_train = [] #linear regression
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0]) #append 1-60 elements
    y_train.append(train_data[i, 0]) #append the 61st value onwards
    if i<= 60:
        print(x_train)
        print(y_train)
        print()
# %%
#convert the x_train and y_train into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# %%
#reshape the data because lstm needs 3d data and now our data is 2d
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1] , 1)) # input the number of steps, time stamps and features, x_train[0]-rows, next cols
x_train.shape
# %%
#Build the LSTM - 4 layers
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1))) #neurons-50
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# %%
#compile the model
model.compile(optimizer='adam', loss='mean_squared_error') #Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent.
model.summary()

# %%
#train the model
model.fit(x_train, y_train, batch_size=1, epochs=1) #fit-train

# %%
#create the testing dataset
#create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60:, :] #index 1543 to 2003
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] #rest of the data -  , :
#y_test will be the values that we want our model to predict
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# %%
#convert the data to numpy array
x_test = np.array(x_test)

# %%
#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #no. of samples, no. of time stamps, no. of features

# %%
#get the models predicted value
predictions = model.predict(x_test)
#as we want exact same values as y_test dataset we use inverse of x_test
predictions = scaler.inverse_transform(predictions)

# %%
#evaluate the model - get the root mean squared error - how accurate is the model
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

# %%
#plot the data
train = data[:training_data_len] #0 to training_data_len
valid = data[training_data_len:] #training_data_len to end
valid['Predictions'] = predictions #giving validation dataset another value of predictions
#visualise the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18) #x-axis
plt.ylabel('Close Price USD ($)', fontsize=18) #y-axis
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']]) #plot the actual data and predicted data
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right') #A legend is used to describe elements for a particular area of a graph.
plt.show()

# %%
#show the valid and predicted prices
valid

# %%
#get the quote for predictions
apple_quote = yf.download('AAPL', start='2012-01-01', end='2020-12-31')
#create new df
dftwo = apple_quote.filter(['Close'])
#get the last 60 days closing price values and convert the dataframe to an array
last_60_days = dftwo[-60:].values
#scale the data into 0 and 1
last_60_days_scaled = scaler.transform(last_60_days) #not using fit as we want the same values from MinMaxScaler
#create an empty list
X_test = []
#append the past 60 days
X_test.append(last_60_days_scaled)
#convert the X_test data into numpy array
X_test = np.array(X_test)
#reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# %%
#get the actual price
apple_quote2 = yf.download('AAPL', start='2021-01-05', end='2021-01-05')
print(apple_quote2['Close'])
