import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import plotting
from datetime import datetime, timedelta
import pandas as pd

st.header("Portfolio Optimization LessgoBaby2.0")
tickers_input = st.sidebar.text_input("Here you can input the tickers of the stocks your portfolio consists (separate with comma):", "")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
st.write("Tickers: ", tickers)

start_date = st.sidebar.date_input("Define the start date for which we should define the efficient portfolio: ")
end_date = datetime.today()
st.write("Dates:", start_date, end_date)

#determining the capital to be allocated
total_value_allocateable = st.sidebar.number_input("Capital to be allocated between stocks (currency is dollars by default): ", )

#determining the maximum weight of an optimal stock
max_weight = st.sidebar.number_input("The maximum weight assigned to each asset in the optimal portfolio: ", )

#inputting the risk-free rate
rf_rate = st.sidebar.number_input("Provide the risk_free rate for the period for more realistic optimization (e.g.: 0.03 = 3%) (use U.S. Treasury yield rates for the relevant period e.g.: 1y, 5y, 10y): ", )

#downloading the dataframe for the calculations
df = yf.download(tickers, start= start_date, end=end_date)["Close"]
st.write("Dataframe:", df)

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
st.write("Expected returns of the stocks: ", mu)
st.write("Covariance matrix of the stocks: ", S)

#Optimizing for maximum Sharpe ratio
ef = EfficientFrontier(mu, S, weight_bounds=(0,max_weight))
raw_weights = ef.max_sharpe(risk_free_rate=rf_rate)
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.csv")
st.write("Cleaned weights:", cleaned_weights)
performance = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)
expected_return_performance = performance[0]
annual_volatility_performance = performance[1]
sharpe_ratio_performance = performance[2]

st.write(f"Expected annual return: {expected_return_performance*100:.2f}%")
st.write(f"Annual volatility: {annual_volatility_performance*100:.2f}%")
st.write(f"Sharpe ratio: {sharpe_ratio_performance}")

# Convert weights dictionary to a pandas DataFrame
weights_df = pd.DataFrame.from_dict(cleaned_weights, orient="index", columns=["Weight"])

# Reset index to make the stock names a column
weights_df.reset_index(inplace=True)
weights_df.rename(columns={"Index": "Stock"}, inplace=True)

# Display the DataFrame in a tabular format
st.write("Portfolio Weights:")
st.dataframe(weights_df)  # Interactive table with sorting and scrolling

#piechart
# Extract tickers and weights with non-zero values
tickers_pie = [ticker for ticker, weight in cleaned_weights.items() if weight > 0]
weights_pie = [weight for weight in cleaned_weights.values() if weight > 0]

# Ensure tickers and weights are aligned
if len(tickers_pie) != len(weights_pie):
    st.write("Error: Mismatch between tickers and weights!")

# Define a function to format the percentage labels
def autopct_format(pct):
    return f'{pct:.1f}%' if pct > 0 else ''

# Plot the pie chart
fig2, ax2 = plt.subplots()
ax2.pie(
    weights_pie,
    labels=tickers_pie,
    autopct=autopct_format,
    startangle=90,
)
ax2.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
ax2.set_title("Portfolio distribution")
st.pyplot(fig2)

latest_prices = get_latest_prices(df)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_value_allocateable)
allocation, leftover = da.greedy_portfolio()

allocation_df = pd.DataFrame.from_dict(allocation, orient="index", columns=["N.o. shares"])
allocation_df.reset_index(inplace=True)
allocation_df.rename(columns={"index": "Stock"}, inplace=True)
st.write("Shares to be allocated:")
st.dataframe(allocation_df)

#st.write("Discrete allocation: ", allocation)
st.write("Funds remaining: ${:.2f}".format(leftover))

#creating the efficient frontier
ef_curve = EfficientFrontier(mu, S, weight_bounds=(0,1))

#generating points for the efficient frontier
fig1, ax = plt.subplots()

#using the annotation for the dots
for i, txt in enumerate(df.columns):
    stock_volatility = S.iloc[i, i] ** 0.5  # Volatility (standard deviation)
    stock_return = mu[i]  # Expected return
    ax.scatter(stock_volatility, stock_return, marker="o", label=None)  # Plot individual asset
    ax.annotate(txt, (stock_volatility, stock_return), textcoords="offset points", xytext=(5,5), ha='center')

#plotting the graph
plotting.plot_efficient_frontier(ef_curve, ax = ax)
ax.scatter(annual_volatility_performance, expected_return_performance , marker="*", s=100, c="r", label='Optimal Portfolio')
ax.set_xlabel("Volatility (Standard Deviation)")
ax.set_ylabel("Expected Return")
ax.legend()

st.subheader("Markowicz Efficient Frontier")
st.pyplot(fig1)



