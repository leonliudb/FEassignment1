import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
import numpy as np
import arch


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
#btc_diff = btc_diff[~np.isnan(btc_diff)]

##7) Estimate an ARMA model (Use the AIC to help determine the proper specification)
res = statsmodels.tsa.stattools.arma_order_select_ic(btc_df, ic=["aic"], trend='nc')
#print(res.aic_min_order)
#(3,2)

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
model=arch.univariate.arch_model(btc_df, vol='Garch', p=1, o=0, q=1, dist='Normal')
results=model.fit()
print(results.summary())

###
###  data_scale_warning.format(orig_scale, rescale), DataScaleWarning
###Iteration:      1,   Func. Count:      6,   Neg. LLF: 71207.53974566457
###Iteration:      2,   Func. Count:     13,   Neg. LLF: 20888.71669716057
###Iteration:      3,   Func. Count:     18,   Neg. LLF: 20888.104838798405
###Iteration:      4,   Func. Count:     23,   Neg. LLF: 20887.631514261735
###Iteration:      5,   Func. Count:     28,   Neg. LLF: 20887.592833441307
###Iteration:      6,   Func. Count:     33,   Neg. LLF: 20887.33879263304
###Iteration:      7,   Func. Count:     38,   Neg. LLF: 20885.95681578927
###Iteration:      8,   Func. Count:     43,   Neg. LLF: 20878.710511108497
###Iteration:      9,   Func. Count:     48,   Neg. LLF: 20841.74461508818
###Iteration:     10,   Func. Count:     53,   Neg. LLF: 20779.234642270203
###Iteration:     11,   Func. Count:     58,   Neg. LLF: 20760.966710315795
###Iteration:     12,   Func. Count:     63,   Neg. LLF: 22123.610883361773
###Iteration:     13,   Func. Count:     69,   Neg. LLF: 3284164.342561654
###Iteration:     14,   Func. Count:     76,   Neg. LLF: 28949137.392824203
###Iteration:     15,   Func. Count:     84,   Neg. LLF: 19797.453422154595
###Iteration:     16,   Func. Count:     89,   Neg. LLF: 19795.167259696238
###Iteration:     17,   Func. Count:     94,   Neg. LLF: 19795.912539007113
###Iteration:     18,   Func. Count:    100,   Neg. LLF: 19794.859303546313
###Iteration:     19,   Func. Count:    105,   Neg. LLF: 19794.85928570096
###Iteration:     20,   Func. Count:    110,   Neg. LLF: 19794.859285480765
###Iteration:     21,   Func. Count:    116,   Neg. LLF: 19794.859304991962
###Optimization terminated successfully    (Exit mode 0)
###            Current function value: 19794.8592788132
###            Iterations: 21
###            Function evaluations: 118
###            Gradient evaluations: 21
###                     Constant Mean - GARCH Model Results
###==============================================================================
###Dep. Variable:                  Close   R-squared:                      -0.892
###Mean Model:             Constant Mean   Adj. R-squared:                 -0.892
###Vol Model:                      GARCH   Log-Likelihood:               -19794.9
###Distribution:                  Normal   AIC:                           39597.7
###Method:            Maximum Likelihood   BIC:                           39620.5
###                                        No. Observations:                 2205
###Date:                Thu, Oct 15 2020   Df Residuals:                     2201
###Time:                        19:48:01   Df Model:                            4
###                                 Mean Model
###============================================================================
###                 coef    std err          t      P>|t|      95.0% Conf. Int.
###----------------------------------------------------------------------------
###mu           516.6533      9.478     54.513      0.000 [4.981e+02,5.352e+02]
###                              Volatility Model
###============================================================================
###                 coef    std err          t      P>|t|      95.0% Conf. Int.
###----------------------------------------------------------------------------
###omega      3.4439e+05  1.312e+04     26.257 5.980e-152 [3.187e+05,3.701e+05]
###alpha[1]       0.9209  5.060e-02     18.199  5.277e-74     [  0.822,  1.020]
###beta[1]    3.5769e-15  5.274e-02  6.782e-14      1.000     [ -0.103,  0.103]
###============================================================================
###
###Covariance estimator: robust
###



#Tried to find out the the proper specification but was not able to resolve this issue

##best_aic = np.inf 
##best_order = None
##best_mdl = None
##
##rng = range(10)
##for i in rng:
##    for j in rng:
##        try:
##            tmp_mdl = statsmodels.tsa.api.ARMA(model, order=(i, j)).fit(method='mle', trend='nc')
##            tmp_aic = tmp_mdl.aic
##            if tmp_aic < best_aic:
##                best_aic = tmp_aic
##                best_order = (i, j)
##                best_mdl = tmp_mdl
##        except: continue
##
##
##print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))