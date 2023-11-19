import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
import tensorflow as tf
from keras.models import  load_model
from sklearn.preprocessing import MinMaxScaler

# Start and the End dates and the stock ticker
start = '2000-01-01'
end = '2022-12-31'
stock_ticker = 'TATAPOWER.NS'

st.title("Stock Market Trend Predictor")
use_input = st.text_input('Enter Stock Ticker', stock_ticker)
if st.button('Analyze'):
    df = pdr.get_data_yahoo(use_input, start)

    #View Data 
    st.subheader("Data from year 2000 to till date:")
    st.dataframe(df.sort_index(ascending=False),use_container_width=True)

    #Plot Graph for Closing Price Vs the Time
    st.subheader("Closing Price VS Time Chart:")
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close,label="Closing Price")
    plt.legend()
    st.pyplot(fig)

    #Plot Graph for Closing Price Vs the Time with 100 Moving Average
    moving_avg_100 = df.Close.rolling(100).mean()
    st.subheader("Closing Price VS Time Chart With 100Moving Average:")
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close, label="Closing Price")
    plt.plot(moving_avg_100,'red', label="100 Moving Average")
    plt.legend()
    st.pyplot(fig)


    #Plot Graph for Closing Price Vs the Time with 100 moving Average and 200 Moving Average 
    moving_avg_200 = df.Close.rolling(200).mean()
    st.subheader("Closing Price VS Time Chart With 100Moving Average and 200Moving Average:")
    fig = plt.figure(figsize=(10,5))
    plt.plot(df.Close, label="Closing Price")
    plt.plot(moving_avg_100,'red', label="100 Moving Average")
    plt.plot(moving_avg_200,'green', label="200 Moving Average")
    plt.legend()
    st.pyplot(fig)

    #Spliting Data in Training and Testing Data 
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    #Scale the training data between 0 and 1
    scaler = MinMaxScaler(feature_range = (0,1))
    data_training_array = scaler.fit_transform(data_training)

    #Load the pre-trained model
    model = load_model('model.h5')

    #Testing Past 
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_test_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100 , input_test_data.shape[0]):
        x_test.append(input_test_data[i-100:i])
        y_test.append(input_test_data[i,0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    #Make Predictions
    y_predicted = model.predict(x_test)

    #Get the scale factor from the scaler and get the original value from the scaled values
    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor

    #Plot Final Graph
    def plot_final_graph():
        st.subheader("Original Stock Price Vs Predicted Stock Price:")
        fig2 = plt.figure(figsize= (12,6))
        plt.plot(y_test, 'blue', label="Original Stock Price")
        plt.plot(y_predicted, 'red', label="Predicted Stock Price")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)


    def main():
        st.title('Stock Price Predictive Analysis')
        
        #Call the function to plot the final graph
        plot_final_graph()
        df_test = pd.DataFrame(y_test, columns=['Original_Price'])
        df_predicted = pd.DataFrame(y_predicted, columns=['Predicted_Price'])
        df_predictions = pd.concat([df_test, df_predicted], axis=1)

        st.subheader("Original and Predicted Stock Price:")
        st.dataframe(df_predictions.sort_index(ascending=False),use_container_width=True, hide_index=True)



    if __name__ == "__main__":
        main()
