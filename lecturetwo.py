#so what I would like to have is an interactive interface with streamlit with which I am able to do some good financial modelling
#I would like to firstly get data for the relevant stocks that I would like to screen
#then create tabs for each individual stock and then through the tabs i will be able to display some important information for the stocks
#then there will be the individual stock weight adjustment for each stock for the portfolio and then it is going to be amazing
#I can implement the empyrial and then also the other quantstats data analysis library with which I can get valuable data for the stocks 
#If I want to I can also add some good portfolio optimization
#but the main task is to get the data from excel and then work with it so it will be sensible to work with 
#now I can grind on 
# I also have to think about how to get dividends data, can I get dividends data from the s&p cap iq pro database or how does that work
#I should also concentrate on dividends and then see which company has payed out the most amounts of dividends from the stock
#I can also input the amount invested in the stock, the amount of shares that we have in each stock, and then there can be a general portfolio description, as with the portfolio geographical parts, then the dividend payouts, then also with the percentages, then we can calculate the weights of the assets, then we can calculate loads of things with this

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from empyrial import empyrial, Engine
import quantstats as qsts
import plotly.express as px
from alpha_vantage.fundamentaldata import FundamentalData 
from stocknews import StockNews
import openai 

st.set_page_config(page_title='Lecture Two For Me', layout= 'wide', page_icon=":smile:")
st.title("Lecture two of the portfolio dashboard")

ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start date')
end_date = st.sidebar.date_input('End Date')

data = yf.download(ticker, start = start_date, end = end_date)
fig = px.line(data, x = data.index, y = data['Close'].squeeze(), title = ticker, labels={'y': 'Close'})
st.plotly_chart(fig)

pricing_data, fundamental_data, news, openai1 = st.tabs(['Pricing Data', 'Fundamental Data', 'News', 'OpenAi ChatGPT'])

with pricing_data: 
    st.write('Price')
    data.columns = ['_'.join(col).strip() for col in data.columns]
    print(data.columns)
    
    data2 = data.copy()
    data2['% Log Returns'] = np.log(data[f"Close_{ticker}"] / data[f"Close_{ticker}"].shift(1))
    data2.dropna(inplace = True)
    st.write(data2)
    
    annual_return = data2['% Log Returns'].mean()*252*100
    st.write("Annual Continuously Compounded Return is:", annual_return, '%')

    stdev = np.std(data2['% Log Returns'])*np.sqrt(252)
    st.write('The Annual Standard Deviation of the stock is:', stdev*100, '%')

    st.write('The Risk Adjusted Return is:', annual_return/(stdev*100))

with news: 
    st.write('News')
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title sentiment: {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News sentiment: {news_sentiment}')


#with fundamental_data: 
#    st.write('Fundamental')
#    key = 'UZL057A5GZFXMXD1'
#    fd = FundamentalData(key, output_format = 'pandas')
#    
#    st.subheader('Balance Sheet')
#    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
#    bs = balance_sheet.T[2:]
#    bs.columns = list(balance_sheet.T.iloc[0])
#    st.write(bs)
#
#    st.subheader('Income Statement')
#    income_statement = fd.get_income_statement_annual(ticker)[0]
#    is1 = income_statement.T[2:]
#    is1.columns = list(income_statement.T.iloc[0])
#    st.write(is1)
#
#    st.subheader('Cash Flow Statement')
#    cash_flow = fd.get_cash_flow_annual(ticker)[0]
#    cfs = cash_flow.T[2:]
#    cfs.columns = list(cash_flow.T.iloc[0])
#    st.write(cfs)


#method = technical_indicator
        #indicator = pd.DataFrame(getattr(ta, method)(low=data[f"Low_{ticker}"], close=data[f"Close_{ticker}"], high=data[f"High_{ticker}"], open=data[f"Open_{ticker}"], volume=data[f"Volume_{ticker}"]))
        #indicator["Close"] = data[f"Close_{ticker}"]
        #figW_ind_new = px.line(indicator)
        #st.plotly_chart(figW_ind_new)
        #st.subheader("Indicator values in a table for the specified period:")
        #st.write(indicator)






