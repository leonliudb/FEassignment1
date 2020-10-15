import yfinance as yf
import pandas as pd


##1) Download daily prices from of Bitcoin (BIT-USD) from Yahoo for period 9/17/2014-9/30/2020.

BTCUSD = yf.Ticker("BTC-USD")
BTCPrices = BTCUSD.history(period='1d', start="2014-09-17", end="2020-09-30")
ClosePrices = BTCPrices['Close']

##2) Load the data into a Pandas data frame.

btc_df = pd.DataFrame(ClosePrices)
#print(btc_df.head(9))


##3) Plot the ACF. What does this suggest about the order of integration?

##4) Apply an augmented Dickey Fuller test. What does the test result tell you about the order of integration?

##5) Estimate the Hurst statistic. What does the test result tell you about the stationarity of the series.

##6) Difference the series using fractional differencing and the d that you estimate.

##7) Estimate an ARMA model (Use the AIC to help determine the proper specification)

##8) Estimate a GARCH model (Use the AIC to help determine the proper specification).