from prophet import Prophet 
import streamlit as st
from datetime import date
import yfinance as yf 
from prophet.plot import plot_plotly
from plotly import graph_objects as go

start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")
stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMZN", "AVGO")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years*365

#@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader("Raw data")
data.columns = ['_'.join(col).strip() for col in data.columns]
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date_'], y=data[f'Open_{selected_stock}'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date_'], y=data[f'Close_{selected_stock}'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting

df_train = data.loc[:, ["Date_", f"Close_{selected_stock}"]].copy()
df_train = df_train.rename(columns={"Date_": "ds", f"Close_{selected_stock}": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)




