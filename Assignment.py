import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
import numpy as np

##1) Download daily prices from of Bitcoin (BIT-USD) from Yahoo for period 9/17/2014-9/30/2020.

BTCUSD = yf.Ticker("BTC-USD")
BTCPrices = BTCUSD.history(period='1d', start="2014-09-17", end="2020-09-30")
ClosePrices = BTCPrices['Close']

##2) Load the data into a Pandas data frame.

btc_df = pd.DataFrame(ClosePrices)
btc_ts = btc_df['Close']
#print(btc_ts.head(9))
#print(btc_df['2018-01-04':'2018-01-06'])

##3) Plot the ACF. What does this suggest about the order of integration?
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(btc_df)
#Non stationary, could be fractional intergrated 

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(btc_df)
#plt.show()


##4) Apply an augmented Dickey Fuller test. What does the test result tell you about the order of integration?
ADF = statsmodels.tsa.stattools.adfuller(btc_ts, maxlag=None, regression='ct', autolag='AIC')
#print(ADF)
#27 lags

##5) Estimate the Hurst statistic. What does the test result tell you about the stationarity of the series.
from hurst import compute_Hc
hurst, s, data = compute_Hc(btc_df, kind='price')
#print(hurst)
#1.19


##6) Difference the series using fractional differencing and the d that you estimate.
d = hurst - 0.5
#print(d)
from fracdiff import Fracdiff
fd = Fracdiff(d)
btc_diff = fd.transform(btc_df)
#print(btc_diff.shape)
btc_diff = btc_diff[~np.isnan(btc_diff)]

##7) Estimate an ARMA model (Use the AIC to help determine the proper specification)
res = statsmodels.tsa.stattools.arma_order_select_ic(btc_diff, ic=["aic"], trend='nc')
#print(res.aic_min_order)#(3,2)
#x = btc_df.to_numpy()
arma = statsmodels.tsa.arima_model.ARIMA(btc_df, order=(3,0,2))
arma_fit = arma.fit(disp=0)
print(arma_fit.summary())
###                              ARMA Model Results
###==============================================================================
###Dep. Variable:                  Close   No. Observations:                 2205
###Model:                     ARMA(3, 2)   Log Likelihood              -15586.776
###Method:                       css-mle   S.D. of innovations            283.895
###Date:                Thu, 15 Oct 2020   AIC                          31187.552
###Time:                        19:14:12   BIC                          31227.442
###Sample:                             0   HQIC                         31202.127
###
###===============================================================================
###                  coef    std err          z      P>|z|      [0.025      0.975]
###-------------------------------------------------------------------------------
###const        4435.1445   2460.975      1.802      0.072    -388.278    9258.567
###ar.L1.Close     1.8071      0.017    105.519      0.000       1.774       1.841
###ar.L2.Close    -1.7730      0.025    -69.584      0.000      -1.823      -1.723
###ar.L3.Close     0.9635      0.017     55.631      0.000       0.930       0.997
###ma.L1.Close    -0.8298      0.019    -44.210      0.000      -0.867      -0.793
###ma.L2.Close     0.9670      0.015     63.417      0.000       0.937       0.997
###                                    Roots
###=============================================================================
###                  Real          Imaginary           Modulus         Frequency
###-----------------------------------------------------------------------------
###AR.1            1.0020           -0.0000j            1.0020           -0.0000
###AR.2            0.4190           -0.9275j            1.0177           -0.1825
###AR.3            0.4190           +0.9275j            1.0177            0.1825
###MA.1            0.4291           -0.9219j            1.0169           -0.1807
###MA.2            0.4291           +0.9219j            1.0169            0.1807
###-----------------------------------------------------------------------------

##8) Estimate a GARCH model (Use the AIC to help determine the proper specification).