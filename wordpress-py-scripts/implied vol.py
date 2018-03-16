def bsm_call_value(S0, K, T, r, sigma):
  #Valuation of European call option in BSM model.
  #Analytical formula.
  
  from math import log, sqrt, exp
  from scipy import stats
  S0 = float(S0)
  d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
  d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
  value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
  # stats.norm.cdf --> cumulative distribution function
  # for normal distribution
  return value

bsm_call_value(S0=10,K=10,T=3./12,r=0.01,sigma=0.1)

# Vega function
def bsm_vega(S0, K, T, r, sigma):
  # Vega of European option in BSM model.
  from math import log, sqrt
  from scipy import stats
  S0 = float(S0)
  d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
  vega = S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(T)
  return vega

bsm_vega(10,10,3./12,0.01,0.1)

# Implied volatility function
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
  for i in range(it):
    sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)
    / bsm_vega(S0, K, T, r, sigma_est))
  return sigma_est

bsm_call_imp_vol(S0=10,K=10,T=3./12,r=0.01,C0=0.3,sigma_est=0.05,it=1000)

import pandas as pd
import os
os.getcwd()

filename = 'vstoxx_data_31032014.h5'

with pd.HDFStore(filename,  mode='r') as h5:
  futures_data = h5.select('futures_data')

with pd.HDFStore(filename,  mode='r') as h5:
  options_data = h5.select('options_data')

#==============================================================================
# futures_data = pd.read_hdf(filename, 'futures_data')
# options_data= pd.read_hdf(filename, 'options_data')
# 
# h5 = pd.HDFStore('vstoxx_data_31032014.h5', 'r')
# futures_data = h5['futures_data'] # VSTOXX futures data
# options_data = h5['options_data'] # VSTOXX call option data
# 
#==============================================================================

h5.close()

options_data.info()
options_data.head()
futures_data.head()
options_data[['DATE', 'MATURITY', 'TTM', 'STRIKE', 'PRICE']].head()

options_data['IMP_VOL'] = 0.0

V0 = 17.6639
r = 0.01
tol = 0.5 # tolerance level for moneyness

for option in options_data.index:
  # iterating over all option quotes
  forward = futures_data[futures_data['MATURITY'] == \
            options_data.loc[option]['MATURITY']]['PRICE'].values[0]
  # picking the right futures value
  if (forward * (1 - tol) < options_data.loc[option]['STRIKE'] < forward * (1 + tol)):
      # only for options with moneyness within tolerance
      imp_vol = bsm_call_imp_vol(V0, # VSTOXX value
                                 options_data.loc[option]['STRIKE'],
                                 options_data.loc[option]['TTM'],
                                 r, # short rate
                                 options_data.loc[option]['PRICE'],
                                 sigma_est=2., # estimate for implied volatility
                                 it=100)
      options_data['IMP_VOL'].loc[option] = imp_vol


futures_data['MATURITY']
options_data.loc[46170]
options_data.loc[46170]['STRIKE']

maturities = sorted(set(options_data['MATURITY']))
maturities

plot_data = options_data[options_data['IMP_VOL'] > 0]

#iterates over all maturities and does the plotting
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16, 12))
for maturity in maturities:
  data = plot_data[options_data.MATURITY == maturity]
  # select data for this maturity
  plt.plot(data['STRIKE'], data['IMP_VOL'],label=maturity.date(), lw=1.5)
  plt.plot(data['STRIKE'], data['IMP_VOL'], 'r.')

plt.grid(True)
plt.xlabel('strike')
plt.ylabel('implied volatility of VSTOXX')
plt.legend()
plt.show()

keep = ['PRICE', 'IMP_VOL']
group_data = plot_data.groupby(['MATURITY', 'STRIKE'])[keep]
group_data

# any aggregation since one element in every group
group_data = group_data.sum()
group_data.head(20)
# check attributes and methods of unknown objects
dir(group_data)
#unique MATURITY and STRIKE
group_data.index.levels










