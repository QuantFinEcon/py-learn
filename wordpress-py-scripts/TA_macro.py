import numpy as np
import pandas as pd
#import pandas_datareader.data as web

#==============================================================================
# get data
#==============================================================================

import quandl
quandl.ApiConfig.api_key = "dDwDihkoAYTL9Tp2rivQ"

#?quandl.get
tickers = ["CHRIS/CME_GC1","CHRIS/CME_CL1","CHRIS/ICE_DX1","FRED/DGS2",
           "YAHOO/INDEX_GSPC","CBOE/VIX"]
data = quandl.get(tickers,start_date="2011-01-01", end_date="2017-01-20")
data.info()
data.tail()
colnames = ["CHRIS/CME_GC1 - Settle", "CHRIS/CME_CL1 - Settle", 
            "CHRIS/ICE_DX1 - Settle", "FRED/DGS2 - VALUE",
            "YAHOO/INDEX_GSPC - Close","CBOE/VIX - VIX Close"]
mydata = data[colnames].fillna(method="ffill") # propagate last non NaN
mydata = mydata.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))) #normalise each column
mydata.info()
mydata.plot(grid=True, figsize=(12, 8)).set(xlabel="Time", ylabel="Cumulative Normalised Return")

#==============================================================================
# alternative plots
#==============================================================================
import seaborn as sns

mydata.plot(subplots=True, layout=(2, 3), figsize=(16, 10), sharex=False)

#iris = sns.load_dataset("iris")
#iris.info()
#iris.tail()
#g = sns.pairplot(iris)
#g1 = sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
#iris.info()


mydata["CBOE/VIX - VIX Close"].hist(bins=100, figsize=(12,8))
fear_idx = mydata['CBOE/VIX - VIX Close'].quantile(.75)
#mydata['CBOE/VIX - VIX Close'].quantile([0.25,0.5,0.75])

ret = mydata.apply(lambda x: (x - x.shift(1)))
ret['Regime'] = np.where(mydata['CBOE/VIX - VIX Close'] >= fear_idx, "fear", "calm")
p1 = sns.pairplot(ret, kind="reg",hue="Regime",markers=["o", "s"])




#==============================================================================
# strategy 1 - long SPX when VIX below 75% percentile of a long history
#==============================================================================
mydata['Regime'] = np.where(mydata['CBOE/VIX - VIX Close'] >= fear_idx, "fear", "calm")
mydata['Regime'].value_counts()
mydata.info()

long_rule = mydata['Regime']  == "calm"

mydata['signal'] = np.where(long_rule , 1, 0)
mydata['signal'].value_counts()

mydata['abs_ret'] = (mydata['YAHOO/INDEX_GSPC - Close'] - mydata['YAHOO/INDEX_GSPC - Close'].shift(1))
mydata['Strategy'] = mydata['signal'].shift(1) * mydata['abs_ret'] # 1 period lag in signal
mydata.info()

# compare buy-and-hold vs strategy
mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(12, 8))
beatBH = mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).tail(1)
print("Strategy performs {0:2.3f}% over buy-and-hold"
      .format(float(beatBH['Strategy']-beatBH['abs_ret'])*100))

#==============================================================================
# strategy 2 - long SPX when USSG2YR above 3 months simple average
#==============================================================================

mydata['YieldTrend'] = mydata['FRED/DGS2 - VALUE'].rolling(window=62,center=False).mean()
long_rule = mydata['FRED/DGS2 - VALUE'] >= mydata['YieldTrend']

mydata['signal'] = np.where(long_rule , 1, 0)
mydata['signal'].value_counts()

mydata['abs_ret'] = (mydata['YAHOO/INDEX_GSPC - Close'] - mydata['YAHOO/INDEX_GSPC - Close'].shift(1))
mydata['Strategy'] = mydata['signal'].shift(1) * mydata['abs_ret'] # 1 period lag in signal
mydata.info()

# compare buy-and-hold vs strategy
mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(12, 8))
beatBH = mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).tail(1)
print("Strategy performs {0:2.3f}% over buy-and-hold"
      .format(float(beatBH['Strategy']-beatBH['abs_ret'])*100))

#==============================================================================
# strategy 3 - long DXY when USSG2YR above 1 month simple average
#==============================================================================

mydata['2yYieldTrend'] = mydata['FRED/DGS2 - VALUE'].rolling(window=21,center=False).mean()
long_rule = mydata['FRED/DGS2 - VALUE'] >= mydata['2yYieldTrend']

mydata['signal'] = np.where(long_rule , 1, 0)
mydata['signal'].value_counts()

mydata['abs_ret'] = (mydata["CHRIS/ICE_DX1 - Settle"] - mydata["CHRIS/ICE_DX1 - Settle"].shift(1))
mydata['Strategy'] = mydata['signal'].shift(1) * mydata['abs_ret'] # 1 period lag in signal
mydata.info()

# compare buy-and-hold vs strategy
mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(12, 8))
beatBH = mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).tail(1)
print("Strategy performs {0:2.3f}% over buy-and-hold"
      .format(float(beatBH['Strategy']-beatBH['abs_ret'])*100))

#==============================================================================
# strategy 4 - long SPX when XAU below 1 month simple average
#==============================================================================

mydata['XAUtrend'] = mydata['CHRIS/CME_GC1 - Settle'].rolling(window=21,center=False).mean()
long_rule = mydata['CHRIS/CME_GC1 - Settle'] <= mydata['XAUtrend']

mydata['signal'] = np.where(long_rule , 1, 0)
mydata['signal'].value_counts()

mydata['abs_ret'] = (mydata['YAHOO/INDEX_GSPC - Close'] - mydata['YAHOO/INDEX_GSPC - Close'].shift(1))
mydata['Strategy'] = mydata['signal'].shift(1) * mydata['abs_ret'] # 1 period lag in signal
mydata.info()

# compare buy-and-hold vs strategy
mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(12, 8))
beatBH = mydata[['abs_ret', 'Strategy']].cumsum().apply(np.exp).tail(1)
print("Strategy performs {0:2.3f}% over buy-and-hold"
      .format(float(beatBH['Strategy']-beatBH['abs_ret'])*100))

