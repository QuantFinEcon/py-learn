import quandl as ql
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
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
    
    
    
    
    
    

# change the end date or otherwise, leave it None
# before 2008 Financial Crisis 2007-2008 
# actually, before Jun 2007 still unaware, fell at Oct 2007
# lets investigate around those dates
yld = ql.get("USTREASURY/YIELD")

# Feb 1990 to now
yld.head(12)
yld.tail(12)

# daily frequency
today = yld.iloc[-1,:]
one_week_ago = yld.iloc[-5,:]
one_month_ago = yld.iloc[-21,:]
two_month_ago = yld.iloc[-41,:]
three_month_ago = yld.iloc[-62,:]
six_month_ago = yld.iloc[-128,:]
one_year_ago = yld.iloc[-261,:]

#ttm= pd.DataFrame([1./12, 3./12, 6./12, 1, 2, 3, 5, 7, 10, 20, 30],
#                  index=today.index, columns=['TTM'])

ans = pd.concat([today,one_week_ago,one_month_ago,
                 two_month_ago,three_month_ago,
                 six_month_ago,one_year_ago], axis = 1)
colnames = ['today', 'one_week_ago','one_month_ago','two_month_ago',
            'three_month_ago','six_month_ago','one_year_ago']
ans.columns = colnames

#plot curve
ans.index = [1./12, 3./12, 1/2, 1, 2, 3, 5, 7, 10, 20, 30]
ax = ans.iloc[:,1:].plot(style={'today':'ro-', 'one_week_ago':'b^-', 'one_month_ago':'gs-',
                'two_month_ago':'mo-', 'three_month_ago':'c^-',
                'six_month_ago':'ys-', 'one_year_ago':'ko-'},
                title=r'U.S. Treasury Yield Curve (Historical)', figsize=(12,8))
ax.set_xlabel('Year to Maturity')
ax.set_ylabel('YTM(%)')


# interpolatation
import scipy.interpolate

#import matplotlib.pyplot as plt
#from scipy import interpolate
#x = np.arange(0, 10)
#y = np.exp(-x/3.0)
#f = interpolate.interp1d(x, y)
#
#xnew = np.arange(0, 9, 0.1)
#ynew = f(xnew)   # use interpolation function returned by `interp1d`
#plt.plot(x, y, 'o', xnew, ynew, '-')
#plt.show()

a=np.asarray(ttm.transpose())[0]
b=np.asarray(ans.iloc[:,1:].transpose()).squeeze()
             
interp = scipy.interpolate.interp1d(a,b,bounds_error=False, fill_value=scipy.nan)
t = np.arange(0,30,6./12)
ans_interp = pd.DataFrame(interp(t).transpose(), index = t, columns = colnames[1:])

#plot interpolated curve
ax=ans_interp.plot(color = ['r','b','g','m','c','y','k'], 
                title=r'U.S. Treasury Yield Curve (Historical), Interpolated', figsize=(12,8))
ax.set_xlabel('Year to Maturity')
ax.set_ylabel('YTM(%)')


'''

# bootstrapping for zero curve
# (1+y[i,j])^(j-i) = (1+s[j])^j / (1+s[i])^i
# (1+y)...(1+y)= (1+s)

today_interp = ans_interp.loc[:,'today']

ans_interp.head(5)
today_interp.head(5)

y = today_interp
# initialise empty df with same shape
s = pd.DataFrame(data=None, columns=["spot_rate"], index=y.index)
# initialise current spot rate as nearest yield rate
s.iloc[0,:] = y[1]

for i in range(2,len(t)-1):
    total = 0
    for j in range(1, i):
      tim = y.index[j]
      total += y.iloc[j,] / (1 + s.iloc[j-1,]) ** tim
    value = ( ((1 + y.iloc[i,]) / (1 - total)) ** (1 / tim) ) - 1
    s.iloc[j,] = value

'''
# contango/backwardation spread against SPX

yld.tail(5)
yld.columns.get_loc("3 MO")
yld[["1 YR","2 YR","5 YR","10 YR","20 YR"]].plot(title="U.S. Treasury (On-Run) Yield to Maturity (%)")

spx = ql.get("GOOG/NYSE_SPY")
spx = spx.Close.to_frame(name="SPY")
spx.tail(5)
 
#==============================================================================
# spread over time - which part of steepening is SPX reacting to?
#==============================================================================

T = spx
for i in yld.columns[:-1]:
  
  base = i
  base_idx = yld.columns.get_loc(base)
  base_next = yld.columns[base_idx+1]
  temp = (yld.iloc[:,base_idx+1]-yld.iloc[:,base_idx]).to_frame(name=base_next + "-" + base)
  T = pd.concat([T,temp], axis=1)
  
T.tail(5)


start_date = "2014-01-01"
T.ix[start_date:].iloc[:,1:].plot(color=tableau20, figsize=(15,10))
T.SPY.ix[start_date:].plot(secondary_y=True,legend=True, mark_right=True, color='black')

T.shape
plot_pairwise(T,start_date,10)

#==============================================================================


base = "1 YR"
base_idx = yld.columns.get_loc(base)
X = yld.iloc[:,(base_idx+1):].sub(yld[base], axis=0)
X = pd.concat([spx,X], axis=1)
X.tail(4)

start_date = "2015-01-01"
X.ix[start_date:].iloc[:,1:].plot(color=tableau20, figsize=(15,10))
X.SPY.ix[start_date:].plot(secondary_y=True,legend=True, mark_right=True, color='black')





def plot_pairwise(data, start_date, total):
    
  NCOL=3
  import math
  NROW=int(math.ceil(total/float(NCOL)))
  
  fig, axes = plt.subplots(nrows=NROW, ncols=NCOL, squeeze=False, figsize=(15,10))
  print( str(data.shape[1]-1) + " charts will be printed")
  colnames = list(data)

  nr=0;nc=0;
  for i in range(1,data.shape[1],1):
      #who1=colnames[i] + "/" + colnames[j]
      #rel=(data[colnames[i]]/data[colnames[j]]).to_frame(name=who1)
      who1 = colnames[i]
      rel = data[who1]
      benchmark=data[colnames[0]].to_frame()
      out=pd.concat([rel,benchmark], axis=1).ix[start_date:]
      
      if nc > (NCOL-1) : nc=0; nr+=1
      
      plt.figure()
      ax = out[[colnames[0],who1]].plot(secondary_y=[colnames[0]], figsize=(6*NCOL,5*NROW), 
              legend=True, ax=axes[nr,nc], sharex=False);
      ax.set_title(who1);
      
      nc+=1
  
  fig.tight_layout()
  






