import quantstats as qs
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import seaborn as sns
from datetime import datetime, timedelta

qs.extend_pandas()

st.header("Portfolio Test LessgoBaby")
tickers_input = st.sidebar.text_input("Here you can input the tickers of the stocks your portfolio consists (separate with comma):", "")
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker (e.g., SPY) (yellow line on graphs):", value="SPY")
method = st.sidebar.selectbox("Select the method of weight calculation: ", ["N.o shares (same currency):", "Stock weights: "])
#start_date_given = st.sidebar.date_input("Select start date:", value=pd.to_datetime("2020-01-01"))
period = st.sidebar.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
prices_df = yf.download(tickers, period="1d")["Close"].iloc[-1]
prices = prices_df.reindex(tickers).values
benchmark = qs.utils.download_returns(benchmark_ticker, period=period)

st.write("Benchark:", benchmark)
st.write("Prices: ", prices)
st.write(f"{benchmark_ticker}")
st.write("Tickers:", tickers)

prices = list(prices)

if method == "N.o shares (same currency):":
    no_shares_input = st.sidebar.text_input("Here you can input the n.o shares in each stock in the right order (has to be the same currency):", "")
    no_shares = [int(shares.strip()) for shares in no_shares_input.split(",") if shares.strip()]
    st.write(no_shares)
    stock_values = [price * shares for price, shares in zip(prices, no_shares)]
    portfolio_value = sum(stock_values)
    st.write(stock_values)
    st.write("Your total portfolio value(in relative currency): ", portfolio_value)
    weights = [float(value) / float(portfolio_value) for value in stock_values]
    weights = list(weights)
    st.write(weights)

else:
    weights_input = st.sidebar.text_input("Here you can specify the weights straightaway (separate with comma, sum to 1):", "")
    weights = [float(weight.strip()) for weight in weights_input.split(",") if weight.strip()]
    weights = list(weights)
    st.write(weights)

#tickers = list(tickers)

portfolio_final = dict(zip(tickers, weights))
st.write(portfolio_final)

st.write("Tickers:", tickers)
st.write("Types of Tickers:", [type(ticker) for ticker in tickers])
st.write("Weights:", weights)
st.write("Types of Weights:", [type(weight) for weight in weights])

print(portfolio_final)
html_file2 = "quanstsportfolio_report.html"
fmaga = portfolio_final_qs = qs.utils.make_index(ticker_weights=portfolio_final, period=period)
st.write("Fmaga:", fmaga)

#help(qs.utils.make_index)
qs.reports.html(fmaga, benchmark, output=html_file2)

st.subheader("QuantStats Portfolio Report")
with open(html_file2, "r", encoding="utf-8") as f:
    html_report = f.read()

st.components.v1.html(html_report, height=1000, width=1200, scrolling=True)

#displaying a correlation matrix of the stocks

st.subheader("Correlation Matrix Heatmap")
end_date = datetime.today()
if period == "1mo": 
    start_date = end_date - timedelta(days=30)
elif period == "3mo":
    start_date = end_date - timedelta(days=90)
elif period == "6mo":
    start_date = end_date - timedelta(days=180)
elif period == "1y":
    start_date = end_date - timedelta(days=365)
elif period == "2y":
    start_date = end_date - timedelta(days=730)
elif period == "5y":
    start_date = end_date - timedelta(days=1825)
elif period == "max":
    start_date = datetime(2000, 1, 1)

data_correlation = yf.download(tickers, start=start_date, end=end_date)["Close"]
returns = data_correlation.pct_change().dropna()
correlation_matrix = returns.corr()

#heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Stock Returns")
plt.show()

#st.dataframe(correlation_matrix)
st.pyplot(plt)

