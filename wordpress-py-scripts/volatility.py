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

#==============================================================================
#  DATA    
#==============================================================================
ql.ApiConfig.api_key = "dDwDihkoAYTL9Tp2rivQ"


tickers = ["CHRIS/CBOE_VX1","CHRIS/CBOE_VX2","CHRIS/CBOE_VX3",
           "CHRIS/CBOE_VX4","CHRIS/CBOE_VX5","CHRIS/CBOE_VX6",
           "CHRIS/CBOE_VX7","CHRIS/CBOE_VX8"]

data = ql.get(tickers)
data = data.filter(regex=r'.Close$', axis=1)
data = data.rename(columns= lambda x: x.replace(r'CHRIS/CBOE_',''))
data = data.rename(columns= lambda x: x.replace(r' - Close',''))
data.tail(5)

spx = ql.get("GOOG/NYSE_SPY")
spx = spx.Close.to_frame(name="SPY")
spx.tail(5)


#plot term structure CCRV
today = data.iloc[-1,:]
one_week_ago = data.iloc[-5,:]
two_week_ago = data.iloc[-10,:]
three_week_ago = data.iloc[-15,:]
one_month_ago = data.iloc[-21,:]
two_month_ago = data.iloc[-41,:]
three_month_ago = data.iloc[-62,:]


ttm= pd.DataFrame(range(1,9,1),index=today.index, columns=['TTM'])
ans = pd.concat([ttm,today,one_week_ago,two_week_ago,three_week_ago,one_month_ago,
                 two_month_ago,three_month_ago], axis = 1)
colnames = ['TTM','today', 'one_week_ago','two_week_ago','three_week_ago',
            'one_month_ago','two_month_ago','three_month_ago']
ans.columns = colnames
ans.index = ["UX" + str(i) for i in range(1,9,1)]
ax = ans.iloc[:,1:].plot(style={'today':'ro-', 'one_week_ago':'b^-', 'two_week_ago':'gs-',
                'three_week_ago':'mo-', 'one_month_ago':'c^-',
                'two_month_ago':'ys-', 'three_month_ago':'ko-'},
                title=r'CBOE SP500 VIX Futures', figsize=(12,8))

ax.set_xlabel('Futures Settlement')
ax.set_ylabel('Implied Volatility(%)')


# plot spread time series
data.ix["2016-01-01":].plot(color=tableau20)

spread = data.sub(data.VX1, axis=0)
colnames = [i + "-VX1" for i in spread.columns][1:]
colnames = ["contango-backwardation"] + colnames[:]
spread.columns = colnames
p1 = pd.concat([spx,spread],axis=1).ix[spread.index[1]:]

start_date = "2013-01-01"
p1.ix[start_date:].iloc[:,1:].plot(color=tableau20, figsize=(15,10))
p1.SPY.ix[start_date:].plot(secondary_y=True,legend=True, mark_right=True, color='black')

# plot ratio time series
s1_3 = (data.VX1/data.VX3).to_frame(name="M1/3")
s3_8 = (data.VX3/data.VX8).to_frame(name="M3/8")
p2 = pd.concat([spx,s1_3,s3_8],axis=1)

start_date = "2015-01-01"
p2.iloc[:,1:].ix[start_date:].plot(figsize=(15,10))
p2.SPY.ix[start_date:].plot(secondary_y=True,legend=True, mark_right=True, color='black')







