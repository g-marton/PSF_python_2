import quantstats as qs
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


qs.extend_pandas()
st.header("Quantstats Programming Trial")
ticker = st.sidebar.text_input("Ticker (blue line on graphs):", value='MSFT')
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker (e.g., SPY) (yellow line on graphs):", value="SPY")
period = st.sidebar.selectbox("Select Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
confidence_interval = st.sidebar.selectbox("Select confidence interval for expected shortfall and VaR:", [0.9, 0.95, 0.99])
frequency = st.sidebar.selectbox("Select for the frequency for VaR and expected shortfall in trading days:", [1, 21, 63, 126, 252])

stock = qs.utils.download_returns(ticker, period = period)
stock2 = qs.utils.download_returns(ticker, period= "5y")
benchmark = qs.utils.download_returns(benchmark_ticker, period=period)

st.write(stock.monthly_returns())

st.subheader("Daily Log Returns:")
st.write(stock)
st.write(benchmark)

st.subheader(f"Key Metrics for {ticker}:")
sharpe = qs.stats.sharpe(stock)
calmar = qs.stats.calmar(stock)
cagr = qs.stats.cagr(stock)
volatility = qs.stats.volatility(stock)
kurtosis = qs.stats.kurtosis(stock)
skewness = qs.stats.skew(stock)
av_return = qs.stats.avg_return(stock)
av_loss = qs.stats.avg_loss(stock)
av_win = qs.stats.avg_win(stock)
win_loss_ratio = qs.stats.win_loss_ratio(stock)
es = qs.stats.expected_shortfall(stock2, confidence=confidence_interval)
es_adjusted = es * (frequency ** 0.5)
VaR = qs.stats.value_at_risk(stock2, confidence=confidence_interval)
VaR_adjusted = VaR * (frequency ** 0.5)
max_drawdown = qs.stats.max_drawdown(stock)


st.write(f"Sharpe ratio for the past {period} period:", sharpe)
st.write(f"Calmar ratio for the past {period} period:", calmar)
st.write(f"The compound annual growth rate for the past {period} period: {cagr*100:.2f}%")
st.write(f"The annualized volatility of the stock based on the past {period} period: {volatility*100:.2f}%")
st.write(f"Kurtosis of daily returns for a {period} period: {kurtosis}")
st.write(f"The skewness of daily returns for a {period} period: {skewness}")
st.write(f"Average return for the past {period} period: {av_return*100:.2f}%")
st.write(f"Average loss for the past {period} period: {av_loss*100:.2f}%")
st.write(f"Average gain for the past {period} period: {av_win*100:.2f}%")
st.write(f"The daily win/loss ratio for the past {period} period: {win_loss_ratio}")
st.write(f"Maximum drawdown for the past {period} period: {max_drawdown*100:.2f}%")
st.write(f"Expected shortfall with {int((confidence_interval) * 100)}% confidence level and {frequency} frequency: {es_adjusted*100:.2f}%")
st.write(f"Value at Risk with {int((confidence_interval) * 100)}% confidence level and {frequency} frequency: {VaR_adjusted*100:.2f}%")
st.write(dir(stock))


st.subheader("Plots:")

fig_cumulative_returns = qs.plots.returns(stock, benchmark, show= False)
st.pyplot(fig_cumulative_returns)
st.write("Where the yellow line corresponds to the benchmark")

fig_log_returns = qs.plots.log_returns(stock, benchmark, show=False)
st.pyplot(fig_log_returns)
st.write("Where the yellow line corresponds to the benchmark")

fig_distribution = qs.plots.distribution(stock, show=False)
st.pyplot(fig_distribution)

fig_drawdown = qs.plots.drawdown(stock, show=False)
st.pyplot(fig_drawdown)

fig_drawdown_periods = qs.plots.drawdowns_periods(stock, show=False)
st.pyplot(fig_drawdown_periods)

fig_earnings = qs.plots.earnings(stock, show=False)
st.pyplot(fig_earnings)

fig_histogram = qs.plots.histogram(stock, show=False)
st.pyplot(fig_histogram)

fig_heatmap = qs.plots.monthly_heatmap(stock, show=False)
st.pyplot(fig_heatmap)

fig_rolling_beta = qs.plots.rolling_beta(stock, benchmark, show=False)
st.pyplot(fig_rolling_beta)

#plotting daily returns
#fig_daily_returns = qs.plots.daily_returns(stock, benchmark, show=False)
#st.pyplot(fig_daily_returns)

fig_rolling_sortino = qs.plots.rolling_sortino(stock, benchmark, show=False)
st.pyplot(fig_rolling_sortino)
st.write("Where the yellow line corresponds to the benchmark")

fig_rolling_sharpe = qs.plots.rolling_sharpe(stock, benchmark, show=False)
st.pyplot(fig_rolling_sharpe)
st.write("Where the yellow line corresponds to the benchmark")


fig_rolling_volatility = qs.plots.rolling_volatility(stock, benchmark, show=False)
st.pyplot(fig_rolling_volatility)
st.write("Where the yellow line corresponds to the benchmark")

fig_snapshot = qs.plots.snapshot(stock, show=False)
st.pyplot(fig_snapshot)

fig_yearly_returns = qs.plots.yearly_returns(stock, benchmark, show=False)
st.pyplot(fig_yearly_returns)




#creating the reports section
st.subheader("Magnificent Reports:")
st.write(dir(qs.reports))

# Generate the QuantStats HTML report
html_file = "quantstats_report.html"
qs.reports.html(stock, benchmark, output=html_file)

# Display the report in Streamlit
st.subheader("QuantStats Report")
with open(html_file, "r", encoding="utf-8") as f:
    html_report = f.read()

st.components.v1.html(html_report, height=1000, width=1200, scrolling=True)

# Remove temporary file if desired
#os.remove(html_file)

#with open (report_file, "r") as f:
#    html_report = f.read()
#st.components.v1.html(html_report, height=800)
#os.remove(report_file)


