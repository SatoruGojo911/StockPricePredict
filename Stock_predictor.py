# IMPORTS

import yfinance as yf 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import streamlit as st
import warnings
import datetime
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

st.header("STOCK PRICE PREDICTOR")
st.write('-'*100)

symbol = st.text_input("Enter the stock symbol you want to predict : ")
st.write('-'*100)
if (symbol):
    stock_symbol = symbol
    start_date = '2010-01-01'
    current_date_time = datetime.datetime.now()
    
    current_date = current_date_time.date()
    yesterday = current_date - datetime.timedelta(days=1)
    formatted_date = current_date.strftime("%Y-%m-%d")
    end_date = str(formatted_date)

    # Download the historical data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    st.write("Fetching data from {} to {}".format(start_date,end_date))
    st.write('-'*100)
    # st.write(stock_data.head())
    X = stock_data.drop('Adj Close', axis=1)
    y = stock_data['Adj Close']
    last_value = stock_data.iloc[-1,:]
    last_close = last_value['Adj Close']
    def plot_stock(col):
        # Create the line plot using Plotly Graph Objects
        fig = go.Figure()
        # Add trace for Adj Close
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[col], mode='lines', name=col))
        # Update layout
        fig.update_layout(title=col, xaxis_title='Date', yaxis_title=col)
        # Show the plot
        st.plotly_chart(fig)
    plot_stock('Open')
    st.write('-'*100)
    plot_stock('Adj Close')
    st.write('-'*100)
    plot_stock('High')
    st.write('-'*100)
    plot_stock('Low')
    st.write('-'*100)
    plot_stock('Volume')
    st.write('-'*100)

    X = stock_data.drop('Adj Close', axis=1)
    y = stock_data['Adj Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    XGB = XGBRegressor()
    SVM = SVR()
    GBR = GradientBoostingRegressor()
    RFR = RandomForestRegressor()

    XGB.fit(X_train,y_train)
    GBR.fit(X_train,y_train)
    RFR.fit(X_train,y_train)
    st.header("PREDICTED VALUE ACCURACIES")
    st.write('-'*100)
    st.write('XBBoosted Regressor Accuracy : {}'.format(XGB.score(X_test,y_test)*100))
    st.write('-'*100)
    st.write('Random Forest Regressor Accuracy : {}'.format(RFR.score(X_test,y_test)*100))
    st.write('-'*100)
    st.write('Gradient Boosted Regressor Accuracy : {}'.format(GBR.score(X_test,y_test)*100))
    ticker = yf.Ticker(stock_symbol)
    live_data = ticker.history(period='1d')
    data_use = live_data.drop(['Dividends','Stock Splits'],axis = 1)
    p_XGB = XGB.predict(data_use)[0]
    p_GBR = GBR.predict(data_use)[0]
    p_RFR = RFR.predict(data_use)[0]
    st.write('-'*100)
    st.header("PREDICTIONS FOR FOLLOWING DAY ")
    st.write('-'*100)
    
    if (p_XGB < last_close):
        st.write('XGBoost predicted value $ {}:red[ ↓ by {} ]'.format(p_XGB,p_XGB-last_close))
    else:
        st.write('XGBoost predicted value $ {}:green[ ↑ by {}] '.format(p_XGB,p_XGB-last_close))
    st.write('-'*100)
    if (p_RFR < last_close):
        st.write('Random Forest predicted value $ {} :red[ ↓ by {} ]'.format(p_RFR,p_RFR-last_close))
    else:
        st.write('Random Forest predicted value $ {} :green[↑ by {}] '.format(p_RFR,p_RFR-last_close))
    st.write('-'*100)
    if (p_GBR < last_close):
        st.write('Gradient Boosted  predicted value $ {} :red[ ↓ by {}] '.format(p_GBR,p_GBR-last_close))
    else:
        st.write('Gradient Boosted  predicted value $ {} :green[↑ by {}] '.format(p_GBR,p_GBR-last_close))

