import quandl as ql
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns

#==============================================================================
# COLOR
#==============================================================================
# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)
    
#==============================================================================
# DATA
#==============================================================================
ql.ApiConfig.api_key = "dDwDihkoAYTL9Tp2rivQ"

tickers = ["GOOG/NYSE_SPY","GOOG/NYSEARCA_XLY","GOOG/NYSE_XLK","GOOG/AMEX_XLI",
           "GOOG/NYSE_XLB","GOOG/AMEX_XLE","GOOG/NYSE_XLP","GOOG/NYSE_XLV",
           "GOOG/NYSE_XLU","GOOG/NYSE_XLF"]

x=ql.get(tickers, start_date = "01-01-2005")
data = x.filter(regex=r'.Close$', axis=1)
data.head(5)

colnames = ['market','discretionary','technology','industrial',
            'materials','energy','staples','healthcare',
            'utilities','financial']
data.columns = colnames
data = data.fillna(method='ffill')
data.tail(5)

##list / column names comprehension
#filter_col = [col for col in list(x) if col.endswith('Close')]
#x.loc[(x == 0).any(axis=1), x.filter(regex=r'.Close$', axis=1).columns].head(5)
#x.loc[(x == 1).any(axis=1), x.filter(regex=r'.Close$', axis=1).columns]
#list(x.filter(regex=r'.Close$', axis=1))
#import re
#[col for col in list(x) if re.search(r'.(Close)$', col)]

## rename colnames
#data = data.rename(columns= lambda x: x.replace(r' - Close',''))
#data = data.rename(columns= lambda x: x.replace(r'GOOG/',''))
#data.head(5)
#
#new_names = dict(zip(map(lambda x: x + " - Close", tickers) ,colnames))
#data = data.rename(columns = new_names)
#data.head(5)



#==============================================================================
# RELATIVE PERFORMANCE
#==============================================================================
X = data.div(data.market,axis='index')
X.tail(5)
# 5 x 5 each plt
ax = X.iloc[:,1:].plot(subplots=True, layout=(3, 3), figsize=(20, 20), sharex=False,color=tableau20)

# discretionary/staple
a=(data['discretionary']/data['utilities']).to_frame(name='discretionary/utilities')
b=data['market'].to_frame()
#a.merge(b).plot()
c=pd.concat([a,b], axis=1).ix['2015-01-01':]
# dual axis
#c['discretionary/staple'].plot(figsize=(15,10),legend=True)
#c['market'].plot(secondary_y=True,legend=True, mark_right=True)
c.plot(secondary_y=['market'], figsize=(15,10), title="1")


#a=5
#b=10
## globals()['a']
#eval('b')

fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(15,10))
c['discretionary/staple'].plot(figsize=(15,10),legend=True, ax=axes[0,0]); axes[0,0].set_title('1');
c['market'].plot(secondary_y=True,legend=True, mark_right=True, ax=axes[0,0]);
c.plot(ax=axes[0,1]); axes[0,1].set_title('2');

sns.set_style("whitegrid", {'axes.grid' : False})

import math
def nCr(n,r):
  f = math.factorial
  return f(n) / f(r) / f(n-r)

total=nCr(9,2)*2
NCOL=3
NROW=int(math.ceil(total/float(NCOL)))

fig, axes = plt.subplots(nrows=NROW, ncols=NCOL, squeeze=False, figsize=(15,10))
data.shape
colnames = list(data)

start_date = '2005-01-03'
nr=0;nc=0;
for i in range(1,data.shape[1]-1,1):
  for j in range(i+1,data.shape[1],1): # print(i,j)
    
    who1=colnames[i] + "/" + colnames[j]
    rel=(data[colnames[i]]/data[colnames[j]]).to_frame(name=who1)
    benchmark=data[colnames[0]].to_frame()
    out=pd.concat([rel,benchmark], axis=1).ix[start_date:]
    
    if nc > (NCOL-1) : nc=0; nr+=1
    
    out[who1].plot(figsize=(6*NCOL,5*NROW),legend=True, ax=axes[nr,nc],sharex=False);
    out[colnames[0]].plot(secondary_y=True, legend=True, 
       mark_right=True, sharex=False, ax=axes[nr,nc]);
    axes[nr,nc].set_title(who1);
    
    nc+=1

# reverse loop
for i in range(data.shape[1]-1,1,-1):
  for j in range(i-1,0,-1): # print(i,j)
    
    who1=colnames[i] + "/" + colnames[j]
    rel=(data[colnames[i]]/data[colnames[j]]).to_frame(name=who1)
    benchmark=data[colnames[0]].to_frame()
    out=pd.concat([rel,benchmark], axis=1).ix[start_date:]
    
    if nc > (NCOL-1) : nc=0; nr+=1
    
    out[who1].plot(figsize=(6*NCOL,5*NROW),legend=True, ax=axes[nr,nc],sharex=False);
    out[colnames[0]].plot(secondary_y=True, legend=True, 
       mark_right=True, sharex=False, ax=axes[nr,nc]);
    axes[nr,nc].set_title(who1);
    
    nc+=1

fig.tight_layout()






#==============================================================================
# BETA ALPHA CORRELATION
#==============================================================================
# https://quantivity.wordpress.com/2011/02/21/why-log-returns/
# data.apply(lambda x: x/x.shift(1)-1 ).tail(5); data.pct_change().tail(5)
ret = data.apply(lambda x: np.log(x/x.shift(1)) )
ret = ret.iloc[1:,:]
ret.shape
ret = ret.loc[(ret!=0).any(axis=1)]
ret.shape

C = np.cov(ret['discretionary'],ret['market'],rowvar=False)
C = np.cov(ret,rowvar=False)
C.shape
C[0,1]/C[1,1]


def calc_beta(df):
    np_array = df.values
    m = np_array[:,0] # market returns are column zero from numpy array
    s = np_array[:,1] # stock returns are column one from numpy array
    covariance = np.cov(s,m) # Calculate covariance between stock and market
    beta = covariance[0,1]/covariance[1,1]
    return beta

# pd.rolling(window=62,center=False).mean() or .apply(.)
def roll(df, w):
    # stack df.values w-times shifted once at each stack
    roll_array = np.dstack([df.values[i:i+w, :] for i in range(len(df.index) - w + 1)]).T
    # roll_array is now a 3-D array and can be read into
    # a pandas panel object
    panel = pd.Panel(roll_array, 
                     items=df.index[w-1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(w), name='roll'))
    # convert to dataframe and pivot + groupby
    # is now ready for any action normally performed
    # on a groupby object
    return panel.to_frame().unstack().T.groupby(level=0)

# http://stats.stackexchange.com/questions/32464/how-does-the-correlation-coefficient-differ-from-regression-slope
# https://en.wikipedia.org/wiki/Beta_(finance)
def beta(df):
    # first column is the market
    X = df.values[:, [0]]
    X = np.concatenate([np.ones_like(X), X], axis=1)
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    # regression: slope=beta 
    return pd.Series(b[1], df.columns[1:], name='Beta')

def alpha(df):
    # first column is the market
    X = df.values[:, [0]]
    X = np.concatenate([np.ones_like(X), X], axis=1)
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    # regression: slope!=beta (correlation if standardised), 
    # intercept=alpha if returns are unstandardised
    return pd.Series(b[0], df.columns[1:], name='Alpha')

# ROLLING Beta
calc_beta(ret.iloc[:,0:2])
beta(ret)
Be = roll(ret,21).apply(beta)
Be.ix['2016-01-01':].plot(figsize=(20,10),color=tableau20)

Be.index
ax = Be.ix['2016-01-01':].plot(subplots=True, layout=(3, 3), 
          figsize=(20, 20), sharex=False,color=tableau20)

# ROLLING alpha
Alp = roll(ret,21).apply(alpha)
Alp.ix['2016-01-01':].plot(figsize=(20,10),color=tableau20)
Alp.ix['2016-01-01':].plot(subplots=True, layout=(3, 3), figsize=(20, 20), sharex=False,color=tableau20)



# ROLLING correlation
ret.corr()
roll_corr = ret.rolling(window=21,center=False).corr(ret['market']).iloc[:,1:]
rc = roll_corr.ix['2016-01-01':]
rc.plot(figsize=(20,10),color=tableau20)
rc.ix['2016-01-01':].plot(subplots=True, layout=(3, 3), figsize=(20, 20), sharex=False,color=tableau20)



#==============================================================================
# Method	Description
# count()	Number of non-null observations
# sum()	Sum of values
# mean()	Mean of values
# median()	Arithmetic median of values
# min()	Minimum
# max()	Maximum
# std()	Bessel-corrected sample standard deviation
# var()	Unbiased variance
# skew()	Sample skewness (3rd moment)
# kurt()	Sample kurtosis (4th moment)
# quantile()	Sample quantile (value at %)
# apply()	Generic apply
# cov()	Unbiased covariance (binary)
# corr()	Correlation (binary) df2.rolling(window=5).corr(df2['one column'])
#==============================================================================

mad = lambda x: np.fabs(x - x.mean()).mean()
ret['market'].rolling(window=60).apply(mad).plot(style='k')






#==============================================================================
# scatter plots
#==============================================================================

ax = ret.plot.scatter(x='market',y='energy',s=50)
ax.axvline(x=0, color = 'k', lw=0.5); ax.axhline(y=0, color = 'k', lw=0.5)

ax = ret.plot.scatter(x='industrial',y='energy',c='market',s=50)
ax.axvline(x=0, color = 'k', lw=0.5); ax.axhline(y=0, color = 'k', lw=0.5)

ax = ret.plot.scatter(x='staples',y='healthcare',c='market',s=ret['market']*1e3)
ax.axvline(x=0, color = 'k', lw=0.5); ax.axhline(y=0, color = 'k', lw=0.5)

## if scatter plot too dense like one line
ret.plot.hexbin(x='staples',y='healthcare')







