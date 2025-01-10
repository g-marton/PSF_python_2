import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import scipy.stats as stats

from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import DiscreteAllocation, get_latest_prices

#opening
st.header("Black Litterman Portfolio Optimization")
st.subheader("This method is really similar to the efficient frontier optimization as we saw earlier, but it takes into account our views of the potential returns for the stocks as well, being more flexible")

#getting risk free rate
rf_rate = st.sidebar.number_input("Provide the risk_free rate for the period for more realistic optimization (e.g.: 0.03 = 3%) (use U.S. Treasury yield rates for the relevant period e.g.: 1y, 5y, 10y): ", )

#determining the capital to be allocated
total_value_allocateable = st.sidebar.number_input("Capital to be allocated between stocks (currency is dollars by default): ", )

#getting maximum weight for a stock
max_weight = st.sidebar.number_input("The maximum weight assigned to each asset in the optimal portfolio: ", )

#getting the tickers
tickers_input = st.sidebar.text_input("Here you can input the tickers of the stocks your portfolio consists (separate with comma):", "")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

#benchmark
benchmark = st.sidebar.text_input("Here you can define the benchmark: ", "")

#getting the dates for the calculations
start_date = st.sidebar.date_input("Define the start date for which we should define the efficient portfolio: ")
end_date = datetime.today()

#downloading stock data
portfolio = yf.download(tickers, start_date, end_date)['Close']
benchmark = yf.download(benchmark, start_date, end_date)['Close']
st.write(benchmark, portfolio)

#downloading market capitalization for each stock in the portfolio
mcaps = {}
for t in tickers: 
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
st.write(mcaps)

#calculating implied market returns
S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(benchmark, risk_free_rate=rf_rate)
st.write("Delta: ", delta)

#heatmap for the covariant correlations
fig1, ax1, = plt.subplots(figsize=(8,6))
sns.heatmap(S.corr(), cmap = 'coolwarm', annot=True, ax=ax1)
st.title("Covariant Correlation Heatmap")
st.pyplot(fig1)

#generating the prior 
market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S, risk_free_rate=rf_rate)
st.write("Market prior:", market_prior)

#horizontal barchart for visualization
fig2, ax2 = plt.subplots(figsize=(10, 5))
market_prior.plot.barh(ax=ax2)
ax2.set_title("Market Prior Compensation")
ax2.set_xlabel("Percentage")
ax2.set_ylabel("Stocks")
st.pyplot(fig2)


#creating a dictionary to integrate views
st.write(tickers)
st.sidebar.subheader("The views on each stock(e.g., 0.10 means the stock will move up 10% in 1 year):")
views = {}
for ticker in tickers:
    views[ticker] = st.sidebar.number_input(f"Expected return for {ticker}:", value=0.0, format="%.2f")
# Display the resulting dictionary
st.write("Your views dictionary:", views)
bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views)

 
confidence_level = st.sidebar.slider(
"Select Confidence Level:",
min_value=0.80,
max_value=0.99,
value=0.95,  # Default to 95%
step=0.01)
# Calculate the z-score corresponding to the confidence level
z_score = stats.norm.ppf((1 + confidence_level) / 2)
# Calculate confidence intervals using log returns
intervals = []
# Compute confidence intervals for each stock in the portfolio
for ticker in tickers:
    # Calculate log returns
    stock_simple_returns = portfolio[ticker].pct_change().dropna()
    # Calculate mean (from views) and standard deviation (from log returns)
    mean_return = views[ticker]
    std_dev = stock_simple_returns.std()
    # Calculate confidence interval using z-score
    lower_bound = mean_return - z_score * std_dev
    upper_bound = mean_return + z_score * std_dev
    # Append to intervals list
    intervals.append((lower_bound, upper_bound))
# Display the confidence intervals
st.write(f"Confidence Intervals ({confidence_level * 100:.0f}%):", intervals)

#calculating omega
variances = []
for lb, ub in intervals:
    sigma = (ub-lb)/2
    variances.append(sigma ** 2)
omega = np.diag(variances)

bl_2 = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta, absolute_views=views, omega=omega)
ret_bl = bl_2.bl_returns()
st.write("Posterior returns? :", ret_bl)

#passing all of this into a dataframe
rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(views)], index=["Prior", "Posterior", "Views"]).T
st.write("Idk what this is", rets_df)

#the other figure
fig3, ax3 = plt.subplots(figsize=(12, 8))
rets_df.plot.bar(ax=ax3)
ax3.set_title("Comparing Compensations Barchart")
st.pyplot(fig3)

#allocating this to a portfolio
S_bl = bl.bl_cov()

st.subheader("Portfolio Allocation")
ef = EfficientFrontier(ret_bl, S_bl, weight_bounds=(0,max_weight))
#ef.add_objective(objective_functions.L2_reg)
raw_weights = ef.max_sharpe(risk_free_rate=rf_rate)
clean_weights = ef.clean_weights()
ef.save_weights_to_file("weights2.csv")
st.write("Cleaned weights:", clean_weights)
performance = ef.portfolio_performance(risk_free_rate=rf_rate, verbose=False)
expected_return_performance = performance[0]
annual_volatility_performance = performance[1]
sharpe_ratio_performance = performance[2]

st.write(f"Expected annual return: {expected_return_performance*100:.2f}%")
st.write(f"Annual volatility: {annual_volatility_performance*100:.2f}%")
st.write(f"Sharpe ratio: {sharpe_ratio_performance}")

# Convert weights dictionary to a pandas DataFrame
weights_df = pd.DataFrame.from_dict(clean_weights, orient="index", columns=["Weight"])

# Reset index to make the stock names a column
weights_df.reset_index(inplace=True)
weights_df.rename(columns={"Index": "Stock"}, inplace=True)

# Display the DataFrame in a tabular format
st.write("Portfolio Weights:")
st.dataframe(weights_df)  # Interactive table with sorting and scrolling

#piechart
# Extract tickers and weights with non-zero values
tickers_pie = [ticker for ticker, weight in clean_weights.items() if weight > 0]
weights_pie = [weight for weight in clean_weights.values() if weight > 0]

# Ensure tickers and weights are aligned
if len(tickers_pie) != len(weights_pie):
    st.write("Error: Mismatch between tickers and weights!")

# Define a function to format the percentage labels
def autopct_format(pct):
    return f'{pct:.1f}%' if pct > 0 else ''

# Plot the pie chart
fig5, ax5 = plt.subplots()
ax5.pie(
    weights_pie,
    labels=tickers_pie,
    autopct=autopct_format,
    startangle=90,
)
ax5.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
ax5.set_title("Portfolio distribution")
st.pyplot(fig5)

latest_prices = get_latest_prices(portfolio)

da = DiscreteAllocation(clean_weights, latest_prices, total_portfolio_value=total_value_allocateable)
allocation, leftover = da.greedy_portfolio()

allocation_df = pd.DataFrame.from_dict(allocation, orient="index", columns=["N.o. shares"])
allocation_df.reset_index(inplace=True)
allocation_df.rename(columns={"index": "Stock"}, inplace=True)
st.write("Shares to be allocated:")
st.dataframe(allocation_df)

#st.write("Discrete allocation: ", allocation)
st.write("Funds remaining: ${:.2f}".format(leftover))

#creating the efficient frontier
ef_curve = EfficientFrontier(ret_bl, S_bl, weight_bounds=(0,max_weight))
#ef_curve.add_objective(objective_functions.L2_reg)

#generating points for the efficient frontier
fig6, ax6 = plt.subplots()

#using the annotation for the dots
for i, txt in enumerate(portfolio.columns):
    stock_volatility = S_bl.iloc[i, i] ** 0.5  # Volatility (standard deviation)
    stock_return = ret_bl[i]  # Expected return
    ax6.scatter(stock_volatility, stock_return, marker="o", label=None)  # Plot individual asset
    ax6.annotate(txt, (stock_volatility, stock_return), textcoords="offset points", xytext=(5,5), ha='center')

#plotting the graph
plotting.plot_efficient_frontier(ef_curve, ax = ax6)
ax6.scatter(annual_volatility_performance, expected_return_performance , marker="*", s=100, c="r", label='Optimal Portfolio')
ax6.set_xlabel("Volatility (Standard Deviation)")
ax6.set_ylabel("Expected Return")
ax6.legend()

st.subheader("Markowicz Efficient Frontier")
st.pyplot(fig6)