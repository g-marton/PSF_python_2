import requests
import pandas as pd
import streamlit as st


base_url = 'https://financialmodelingprep.com/api'

st.header('Financial Modeling Prep Stock Screener')
ticker = st.sidebar.text_input('Ticker:', value='MSFT')
financial_data = st.sidebar.selectbox('Financial Data Type:', options = ('income-statement', 'balance-sheet-statement', 
                                                                         'cash-flow-statement', 'historical-price-full/stock_dividend', 
                                                                         'income-statement-growth', 'balance-sheet-statement-growth', 'cash-flow-statement-growth'
                                                                         'key-metrics', 'key-metrics-ttm', 'ratios-ttm', 'ratios', 'financial-growth', 
                                                                         'quote', 'enterprise-values', 'rating', 'historical-rating', 'discounted-cash-flow', 
                                                                         'historical-discounted-cash-flow-statement', 'historical-price-full', 'historical price smaller intervals'))

if financial_data == 'historical price smaller intervals': 
    interval = st.sidebar.selectbox('Interval', options=('1min', '5min', '15min', '30min', '1hour', '4hour'))
    financial_data = 'historical-chart/'+interval

API_KEY = 'vSYSW0hsqQXc5N32P4jMXXuv0pOqPwXe'

url = f'{base_url}/v3/{financial_data}/{ticker}?apikey={API_KEY}'


response = requests.get(url)
data = response.json()
df = pd.DataFrame(data).T

st.write(df)
st.write(data)

if financial_data == 'historical-price-full/stock_dividend':
    st.header("Dividend calculator for the stock")
    start_date_dividend = st.date_input("Please select the starting date from which we should add up dividends:")
    historical_dividends = data.get('historical', [])

    if not historical_dividends:
        st.write('No dividend data available for the selected stock.')
    else:
        # Convert to DataFrame
        df_2 = pd.DataFrame(historical_dividends)

        # Filter by Start Date
        df_2['date'] = pd.to_datetime(df_2['date'])
        filtered_dividends = df_2[df_2['date'] >= pd.to_datetime(start_date_dividend)]

        # Sum Dividends
        total_dividends = filtered_dividends['adjDividend'].sum()

        # Display Results
        st.write(f'Total Dividends Received for {ticker} since {start_date_dividend}: ${total_dividends:.2f}')
        st.write('Dividend Details:')
        st.write(filtered_dividends[['date', 'adjDividend', 'dividend']])








