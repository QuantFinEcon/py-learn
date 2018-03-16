import numpy as np
import pandas as pd
import pandas_datareader.data as web


sp500 = web.DataReader('^GSPC', data_source='yahoo',start='1/1/2007', end='1/30/2017')
sp500.info()

sp500['Close'].plot(grid=True, figsize=(16, 10))

sp500['3m'] = np.round(sp500['Close'].rolling(window=62,center=False).mean(), 2)
sp500['1y'] = np.round(sp500['Close'].rolling(window=252,center=False).mean(), 2)
sp500[['Close', '3m', '1y']].tail()

sp500[['Close', '3m', '1y']].plot(grid=True, figsize=(16, 10))

sp500['3m-1y'] = (sp500['3m'] - sp500['1y'])/sp500['1y']
sp500['3m-1y'].tail()
sp500['3m-1y'].head(252)
sp500['3m-1y'].hist(bins=20, figsize=(12,8))
sp500['3m-1y'].std()
sp500.info()

mult = 0.5
SD = sp500['3m-1y'].std() * mult # stdDev signal threshold FILTER
sp500['Regime'] = np.where(sp500['3m-1y'] > SD, 1, 0)
sp500['Regime'] = np.where(sp500['3m-1y'] < -SD, -1, sp500['Regime'])
sp500['Regime'].value_counts()
sp500['Regime'].plot(lw=1.5, ylim=(-1.1, 1.1))

sp500['log_ret'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500['Strategy'] = sp500['Regime'].shift(1) * sp500['log_ret'] # 1 period lag in signal
sp500.info()

# compare buy-and-hold vs strategy
sp500[['log_ret', 'Strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(12, 8))
beatBH = sp500[['log_ret', 'Strategy']].cumsum().apply(np.exp).tail(1)
print("Strategy with {0:1.1f} SD performs {1:2.3f}% over buy-and-hold"
      .format(mult,float(beatBH['log_ret']-beatBH['Strategy'])))

