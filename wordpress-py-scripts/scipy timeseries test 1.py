import numpy as np
import pandas as pd
import statsmodels.api as sm
tsa = sm.tsa # as shorthand


mdata = sm.datasets.macrodata.load().data
type(mdata)

endog = np.log(mdata['m1'])
exog = np.column_stack([np.log(mdata['realgdp']), np.log(mdata['cpi'])])
exog = sm.add_constant(exog, prepend=True)
exog
res1 = sm.OLS(endog, exog).fit()

acf, ci, Q, pvalue = tsa.acf(res1.resid, nlags=4,alpha=.05, qstat=True,unbiased=True)
acf
pvalue

tsa.pacf(res1.resid, nlags=4)








#==============================================================================
# FILTER
#==============================================================================

from scipy.signal import lfilter
data = sm.datasets.macrodata.load()
infl = data.data.infl[1:]
data.data.shape

# get 4 qtr moving average
infl = lfilter(np.ones(4)/4, 1, infl)[4:]
unemp = data.data.unemp[1:]

#To apply the Hodrick-Prescott filter to the data 3, we can do
infl_c, infl_t = tsa.filters.hpfilter(infl)
unemp_c, unemp_t = tsa.filters.hpfilter(unemp)

#The Baxter-King filter 4 is applied as
infl_c = tsa.filters.bkfilter(infl)
unemp_c = tsa.filters.bkfilter(unemp)

#The Christiano-Fitzgerald filter is similarly applied 5
infl_c, infl_t = tsa.filters.cfilter(infl)
unemp_c, unemp_t = tsa.filters.cfilter(unemp)


#plot
INFLA=pd.DataFrame(infl_c,columns=['INFLA'])
UNEMP=pd.DataFrame(unemp_c[4:],columns=['UNEMP'])
pd.concat([INFLA,UNEMP],axis=1).plot()

INFLA=pd.DataFrame(infl_t,columns=['INFLA'])
UNEMP=pd.DataFrame(unemp_t[4:],columns=['UNEMP'])
pd.concat([INFLA,UNEMP],axis=1).plot()


#==============================================================================
# BENCHMARKING TO STANDARDISE LOWER FREQ TO HIGHER FREQ
#==============================================================================
iprod_m = np.array([ 87.4510, 86.9878, 85.5359, #INDUSTRIAL PRODUCTION INDEX
                    84.7761, 83.8658, 83.5261, 84.4347,
                    85.2174, 85.7983, 86.0163, 86.2137,
                    86.7197, 87.7492, 87.9129, 88.3915,
                    88.7051, 89.9025, 89.9970, 90.7919,
                    90.9898, 91.2427, 91.1385, 91.4039,
                    92.5646])
gdp_q = np.array([14049.7, 14034.5, 14114.7,14277.3, 14446.4, 14578.7, 14745.1,14871.4])
gdp_m = tsa.interp.dentonm(iprod_m, gdp_q,freq="qm")


a=[]
[a.extend([i]*4) for i in gdp_q]

x=pd.DataFrame([iprod_m,gdp_m],index=['IPROD','GDP MONTHLY']).T
x.plot(secondary_y='IPROD')
pd.DataFrame([gdp_m,a],index=['monthly','quarterly']).T.plot(secondary_y='quarterly')


mdata = sm.datasets.macrodata.load().data
mdata = mdata[['realgdp','realcons','realinv']]
names = mdata.dtype.names
data = mdata.view((float,3))
data = np.diff(np.log(data), axis=0)
#statsmodels.tsa.vector_ar.var_model.VAR
import statsmodels
data
model = statsmodels.tsa.vector_ar.var_model.VAR(data)
model.endog_names = names
res = model.fit(maxlags=2)
res.plot_forecast(5)
res.fevd().plot()

#autocorrelation
res.plot_sample_acorr()
# impulse response
irf = res.irf(10) # 10 periods
irf.plot()

# granger causality with VAR.fit
res.test_causality(equation='y1',variables=['y2'])
res.test_causality(equation='y1',variables=['y2','y3'])










