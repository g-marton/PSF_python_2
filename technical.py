import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import pandas_ta as ta

st.set_page_config(page_title='Lecture Three For Me', layout= 'wide', page_icon=":smile:")
st.title("Lecture three of the portfolio dashboard")

ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start date')
end_date = st.sidebar.date_input('End Date')

data = yf.download(ticker, start = start_date, end = end_date)
fig = px.line(data, x = data.index, y = data['Close'].squeeze(), title = ticker, labels={'y': 'Close'})
st.plotly_chart(fig)

pricing_data, fundamental_data, news, openai1, technical = st.tabs(['Pricing Data', 'Fundamental Data', 'News', 'OpenAi ChatGPT', 'Technical Analysis Dashboard'])

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

with technical: 
    st.subheader('Technical Analysis Dashboard:')
    df= pd.DataFrame()
    indicator_list=df.ta.indicators(as_list=True)
    st.write(indicator_list)
    technical_indicator = st.selectbox('Tech Indicator', options=indicator_list)
    method = technical_indicator
    indicator = pd.DataFrame(getattr(ta, method)(low=data[f"Low_{ticker}"], close=data[f"Close_{ticker}"], high=data[f"High_{ticker}"], open=data[f"Open_{ticker}"], volume=data[f"Volume_{ticker}"]))
    indicator["Close"] = data[f"Close_{ticker}"]
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)



