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


from plotly import graph_objs as go




st.title('Stock Market Price Prediction')

stocks = ("AAPL" , "GOOGL" , "MSFT" ,"GME" ,"AMZN","NVDA","FB")
selected_stocks = st.selectbox("Select dataset for prediction",stocks)

# n_years = st.slider("Years of prediction" , 1, 20)
# period = n_years * 365


# @st.cache
# def load_data(ticker):
#     data = yf.download(ticker , start , end)
#     data.reset_index(inplace=True)
#     return data


# data_load_state = st.text("Load data...")
# data = load_data(selected_stocks)
# data_load_state.text("Loading data...done!")

# st.subheader('Raw data')
# st.write(data.tail())




df = data.DataReader(selected_stocks , 'yahoo', start , end)

st.subheader('Data from 2010 - 2022')

st.write(df.describe())

# st.subheader('Closing price vs Time chart')
# fig = plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# st.pyplot(fig)


# st.subheader('Volume Traded')
# fig = plt.figure(figsize=(8,4))

# plt.plot(df.Volume , 'b')
# plt.fill(df.Volume , 'b')
# st.pyplot(fig)


#Spliting the data into training and testing 


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])

data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
#Splitting data into x_train and y_train



#load my model

model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing,ignore_index=True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# #Final graph
# st.subheader('Prediction vs Original')
# fig2 = plt.figure(figsize = (12 ,6))
# plt.plot(y_test, 'b', label = 'Original Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)




ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# fig3 = plt.figure(figsize = (15 ,8))
# # plt.plot(y_test, 'b', label = 'Original Price')
# # plt.plot(df.Close, 'b', label = 'Predicted Price')
# plt.plot(df.Close)
# # plt.plot(ma100, 'g')
# # plt.plot(ma200, 'r')
# plt.xlabel('Time')
# plt.plot(ma100)
# plt.plot(ma200)



# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig3)

today = round(df.Close[-1])


if ma100[-1] > ma200[-1]:
    st.subheader('BULLISH')
else:
    st.subheader('BEARISH')



def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df.Close,mode='lines', name='Stock Price'))
    
    fig.add_trace(go.Scatter(y=ma200, name='Moving Average 200',))
    fig.add_trace(go.Scatter(y=ma100, name='Moving Average 100'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()



df1 = df[2450:]



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


# figg = plt.figure(figsize=(15, 6))
# rsi_ewma.plot()

# plt.legend(['RSI via EWMA', 'RSI via SMA', 'RSI via RMA/SMMA/MMA (TradingView)'])
# st.pyplot(figg)

def plot_rsi_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rsi_ewma, name='RSI'))
    fig.add_hline(y=80 ,line_color="green" ,opacity=0.1)
    fig.add_hline(y=20 ,line_color="green" ,opacity=0.1)
    fig.add_hrect(y0=20, y1=80, line_width=0, fillcolor="green", opacity=0.05)
    fig.layout.update(title_text="Relative Strength Index", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_rsi_data()

if rsi_ewma[-1] > 80:
    st.subheader('Stock is Overpriced')
if (rsi_ewma[-1] < 20):
    st.subheader('Stock is Unver-Valued')
else:
    st.subheader('Stable')





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

st.subheader('MACD DATA')

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

st.write(aapl_macd.describe())

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
# plot_macd(data['Close'], aapl_macd['macd'], aapl_macd['signal'], aapl_macd['hist'])


