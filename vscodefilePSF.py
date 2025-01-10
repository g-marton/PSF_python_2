import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from empyrial import empyrial, Engine
from quantstats import reports

st.set_page_config(page_title="PSF Dashboard", layout= "wide")
st.title("Prosper Social Finance Portfolio Analysis Dashboard")
st.write("The content and the explanation will be written here")

tickers = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, GOOGL")
start_date = st.sidebar.date_input("Select default start date:", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Select end date:", value=pd.to_datetime("2023-12-31"))

df = pd.DataFrame({
    'first column': [1,2,3,4],
    'second column': [5,6,7,8]})

st.write(df)
st.sidebar.slider('anya', 0, 100)

if st.sidebar.checkbox('Balloons'):
    st.balloons()

st.sidebar.date_input('Date')
st.sidebar.number_input('Number input', 0, 100)
option = st.sidebar.selectbox('Which politician is the best?', ['Orbán', "Putyin", 'Trump'])

chart_data = pd.DataFrame(
    np.random.randn(100, 3),
    columns = ['a', 'b', 'c'])

st.line_chart(chart_data)

tab1, tab2, tab3 = st.tabs(['First tab', 'Second Tab', 'Third Tab'])
with tab1: 
    st.header('Vamos Alaplajja')
with tab2: 
    st.header('Hakunamatata')
with tab3: 
    st.header('Bomboclat')

import time
last_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    last_iteration.text(f'iteration {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)
