import quandl as ql
ql.ApiConfig.api_key = "dDwDihkoAYTL9Tp2rivQ"
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
    
    
#PAYEMS = ql.get("FRED/PAYEMS")
#PAYEMS.info
#PAYEMS.iloc[:,0]
#PAYEMS.index[0]
#NFP = PAYEMS.apply(lambda x: (x - x.shift(1)))
#NFP.ix['2006':].plot()
                  
                
            
#==============================================================================
# helpers
#==============================================================================

import calendar

def fix_dates(data):
  dates = data.index
  data['DT'] = data.index
  
  for j in range(0,len(dates)):
    i = data['DT'].iloc[j]
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    year=i.year; month=i.month
    monthcal = c.monthdatescalendar(year,month)
    # first friday of month
    t = pd.Timestamp([day for week in monthcal for day in week if \
                    day.weekday() == calendar.FRIDAY and \
                    day.month == month][0])
    # US timezone 08:30
    t = t.replace(hour=8, minute=30) 
    data['DT'].iloc[j] = t
  
  data = data.set_index(['DT'])
  return(data)

#fix_dates(PAYEMS)



#==============================================================================
# get data, validate
#==============================================================================

tickers = ['PAYEMS','CES0500000003','CES0500000011','AWHAETP','MANEMP','CEU4200000001','USCONS','USGOVT','USEHS','USPBS','CEU5500000001','USMINE','CEU4142000001','USINFO','CES4300000001','USLAH']
tickers = ["FRED/" + i for i in tickers]
coln = ['NFP','Average Hourly Earnings of Total Private','Average Weekly Earnings of Total Private','Average Weekly Hours of Total Private','Manufacturing','Retail Trade','Construction','Government','Education and Health Services','Professional and Business Services','Financial Activities','Mining and logging','Wholesale Trade','Information Services','Transportation and Warehousing','Leisure and Hospitality']
data=ql.get(tickers)
backup=data
#data=backup
data.columns = coln
data=fix_dates(data)
#change for NFP, earnings, hours
#data.iloc[:,0]=data.iloc[:,0].to_frame().apply(lambda x: (x - x.shift(1)))

#hindsight
#data = data.iloc[0:len(data)-2,:]

net_earnings_chg = data.iloc[:,[1,2,3]]     
net_earnings_chg = (data.iloc[:,1]*data.iloc[:,3]/data.iloc[:,2]).to_frame()
net_earnings_chg.dropna(axis=0).plot()
data = data.drop(data.columns[[1,2,3]],axis=1)
data.tail(1).T

         
#==============================================================================
# stacked area over time
#==============================================================================
p=data.iloc[:,1:]
p.tail(1).T
p.plot().legend(loc='center left', bbox_to_anchor=(1, 0.5))
p.plot.area().legend(loc='center left', bbox_to_anchor=(1, 0.5))
perc=p.apply(lambda c: c / c.sum() * 100, axis=1)
#perc.plot.area().legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax = perc.plot.area()
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc='center left',bbox_to_anchor=(1, 0.5)) 


#==============================================================================
# total NFP attribution by sector
#==============================================================================

p2=p.apply(lambda x: (x - x.shift(1)))
p2.plot().legend(loc='center left', bbox_to_anchor=(1,0.50))
#contribution
ax = p2.ix['2015':].plot(kind='bar', stacked=True)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc='center left',bbox_to_anchor=(1, 0.5))
ax.axhline(y=0,color='black',linewidth=0.5)
#import inspect
#inspect.getargspec(p.plot().legend)

#==============================================================================
# by month for each sector
#==============================================================================

len(p2)/12
p2.tail(1).T
tit = ['Retail Trade','Financial Activities','Wholesale Trade','Manufacturing','Mining and logging']
k=4
pp=p2[tit[k]]
pp.plot(title=tit[k])

gg=pp.groupby([(pp.index.month)])
ma = map(lambda i: calendar.month_abbr[i],gg.groups.keys())

#http://jonathansoma.com/lede/data-studio/classes/small-multiples/long-explanation-of-using-plt-subplots-to-create-small-multiples/
plt.figure(figsize=(22,18), facecolor='white').\
          suptitle(tit[k],fontsize='15',fontweight="bold")
for k,v in gg:
  print k
  print v.to_frame().head(5)
  ax = plt.subplot(6,2,k)
  v.to_frame().plot(kind='bar',title=ma[k-1], figsize=(11, 9), legend=False, ax=ax)
  ax.get_xaxis().set_visible(False)
  plt.axhline(y=0,linewidth=0.5,color='red')
  #  ax.set_axis_off()
plt.tight_layout()


#==============================================================================
# absolute
#==============================================================================

#GB=DF.groupby([(DF.index.year),(DF.index.month)]).sum()
g=p2.groupby([(p2.index.month)])
avr=g.mean()
avr.plot(kind='bar').legend(loc='center left', bbox_to_anchor=(1, 0.5))
avr.T.plot(kind='bar',figsize=(10,20)).legend(loc='center left', bbox_to_anchor=(1, 0.5))


#==============================================================================
# percentage
#==============================================================================
p3=p.dropna(axis=0)
p3=p3.apply(lambda x: ((x/x.shift(1))-1)*100, axis=0)

g=p3.groupby([(p3.index.month)])
avr=g.mean()
avr.plot(kind='bar').legend(loc='center left', bbox_to_anchor=(1, 0.5))
avr.T.plot(kind='bar').legend(loc='center left', bbox_to_anchor=(1, 0.5))


#==============================================================================
# groupby <month,sector>
#==============================================================================

p4=p2.stack()
p4.plot()

#==============================================================================
# test corr, pre-post relation or lead lag
#==============================================================================

DXY = ql.get('FRED/TWEXM')
DXY.plot()


#==============================================================================
# FORECASTING
#==============================================================================
#==============================================================================
# each sector business outlook ISM, Philly Fed Manufacturing survey index
# jobless claims, ADP 2 days before NFP
# RETAIL SALES, PCE aka CPI via personal income report
# GDP growth estimates
#==============================================================================
https://www.federalreserve.gov/monetarypolicy/openmarket.htm
http://www.integritas.asia/nfp/
http://www.businessinsider.com/april-nfp-vs-latest-philly-fed-report-2012-4?IR=T&r=US&IR=T
https://www.bea.gov/newsreleases/national/pi/pinewsrelease.htm
http://www.investopedia.com/terms/p/pce.asp
https://www.census.gov/retail/index.html

fact = ['NPPTTL','JTSJOL','RSXFS','USSLIND','MNFCTRSMSA','CFNAIDIFF','ICSA','GACDFSA066MSFRBPHI']
fact = ["FRED/" + i for i in fact]
coln = ['ADP NFP','Job Openings','Core Retail Sales','Leading Index','Manufacturers Sales','Chicago Diffusion Index','Jobless Claims','Philly Fed Index']
data2=ql.get(fact)
backup2=data2
data2.columns = coln

data2.tail(20)











