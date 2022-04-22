import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from keras.models import load_model
import streamlit as st
from sqlite3 import Date
import pandas_datareader as data

from plotly import graph_objs as go


from signal import signal
from sqlite3 import Date
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf 
import os
import numpy as np
import pandas as pd
import random



st.title('Stock Market Price Prediction')

stocks = ("AAPL" , "GOOGL" , "MSFT" ,"GME" ,"AMZN","NVDA","FB")
selected_stocks = st.selectbox("Select dataset for prediction",stocks)

model = load_model('sayan_model.h5')

print(model.summary())

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
   
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result


import time
from tensorflow.keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 15
# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")
### model parameters
N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False
### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 10
# Amazon stock market
ticker = selected_stocks

data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS)


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

future_price = predict(model, data)



st.subheader(f'The Future price of {selected_stocks} after 15 days is ${future_price:.2f}')


from signal import signal
from sqlite3 import Date
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf 
start = '2010-01-01'
end = Date.today().strftime("%Y-%m-%d")




# st.title('Stock Market Price Prediction')

# stocks = ("AAPL" , "GOOGL" , "MSFT" ,"GME" ,"AMZN","NVDA","FB")
# selected_stocks = st.selectbox("Select dataset for prediction",stocks)



df = data.DataReader(selected_stocks , 'yahoo', start , end)

st.subheader('Data from 2010 - 2022')

st.write(df.describe())


df1 = df[2450:]


ma100 = df1.Close.rolling(21).mean()
ma200 = df1.Close.rolling(50).mean()


#MOVING AVERAGE

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df1.Close,mode='lines', name='Stock Price'))
    
    fig.add_trace(go.Scatter(y=ma200, name='Moving Average 200',))
    fig.add_trace(go.Scatter(y=ma100, name='Moving Average 100'))
    fig.layout.update(title_text="SIMPLE MOVING AVERRAGE", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()




#RSI
# Get data
window_length = 14

# Get just the adjusted close
close = df1['Close']
# Get the difference in price from previous step
delta = close.diff()
# Get rid of the first row, which is NaN since it did not have a previous 
# row to calculate the differences
delta = delta[1:] 

# Make the positive gains (up) and negative gains (down) Series
up, down = delta.clip(lower=0), delta.clip(upper=0).abs()

# Calculate the RSI based on EWMA
# Reminder: Try to provide at least `window_length * 4` data points!
roll_up = up.ewm(span=window_length).mean()
roll_down = down.ewm(span=window_length).mean()
rs = roll_up / roll_down
rsi_ewma = 100.0 - (100.0 / (1.0 + rs))

def plot_rsi_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rsi_ewma, name='RSI'))
    fig.add_hline(y=80 ,line_color="green" ,opacity=0.1)
    fig.add_hline(y=20 ,line_color="green" ,opacity=0.1)
    fig.add_hrect(y0=20, y1=80, line_width=0, fillcolor="green", opacity=0.05)
    fig.layout.update(title_text="RELATIVE SRENGTH INDEX", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_rsi_data()




#BOLLINGER BAND




def get_sma(prices, rate):
    return prices.rolling(rate).mean()

def get_bollinger_bands(prices, rate=20):
    sma = get_sma(prices, rate)
    std = prices.rolling(rate).std()
    bollinger_up = sma + std * 2 # Calculate top band
    bollinger_down = sma - std * 2 # Calculate bottom band
    return bollinger_up, bollinger_down

df1.index = np.arange(df1.shape[0])
closing_prices = df1['Close']

bollinger_up, bollinger_down = get_bollinger_bands(closing_prices)


def plot_bollinger_data():
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=closing_prices, name='Original'))
    fig3.add_trace(go.Scatter(y=bollinger_up, name='Up'))
    fig3.add_trace(go.Scatter(y=bollinger_down, name='Down'))

    fig3.layout.update(title_text="BOLLINGER BAND", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

plot_bollinger_data()


#MOVING AVERAGE CONVERGENCE DIVERGENCE

def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

aapl_macd = get_macd(df1['Close'], 26, 12, 9)

# st.write(aapl_macd.describe())

def plot_macd_data():
    fig3 = go.Figure()
    fig3.add_hline(y=0 ,line_color="black", opacity=0.5)
    fig3.add_hrect(y0=0, y1=10, line_width=0, fillcolor="green", opacity=0.1)
    fig3.add_hrect(y0=0, y1=-10, line_width=0, fillcolor="red", opacity=0.1)

    fig3.add_trace(go.Scatter(y=aapl_macd['macd'], name='Macd'))
    fig3.add_trace(go.Scatter(y=aapl_macd['signal'], name='Signal'))
    fig3.add_trace(go.Scatter(y=aapl_macd['hist'], name='hist'))

    fig3.layout.update(title_text="MOVING AVERAGE CONVERGENCE DIVERGENCE", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

plot_macd_data()



readData = pd.read_csv('fundamentals.csv')
dfF = pd.DataFrame(readData)
print(readData)
dfF= dfF.set_index('symbol')
st.write(dfF.describe())

fundamentals = ['dividendYield' , 'marketCap', 'beta' ,'forwardPE','revenueGrowth','targetHighPrice','bookValue','fiftyDayAverage']

dfF = dfF[dfF.columns[dfF.columns.isin(fundamentals)]]

# st.write(dfF.describe())

st.bar_chart(dfF.dividendYield)
st.bar_chart(dfF.forwardPE)
st.bar_chart(dfF.bookValue)
st.bar_chart(dfF.marketCap)
