import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
import sys
from datetime import date
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import itertools
import inspect 
import re


import seaborn as sns
sns.set(style="white")

#from arcapital import *
#setup_bbg()
#
#import arcapital
#arcapital.setup_bbg()
#
#import os
#os.system("arcapital.py")

#==============================================================================
# GET TICKERS FROM BBG
#==============================================================================
filename = 'AxJ ICB Sector Breakdown.xlsm'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Airlines')
#print the column names
print df.columns

#get a data frame with selected columns
FORMAT = ['Ticker']
CT = df[FORMAT]
for i in CT: print CT[i]


#==============================================================================
# GET FLDS FROM BBG
#==============================================================================

#filename = 'risk tickers.xlsx'
#xls = pd.ExcelFile(filename)
#sht = xls.sheet_names
#for i in sht: print i
#
#df = pd.read_excel(filename,sheetname='Fundamental')
##print the column names
#print df.columns
#
##get the values for a given column
#FLDS = df['Metric'].values
#for i in FLDS: print i
#FLDS.shape
#FLDS = FLDS.tolist()
#FLDS = [x for x in FLDS if str(x) != 'nan']


FLDS = ["EBITDA","BEST_EBITDA","IS_EPS","BEST_EPS","CUR_MKT_CAP"]
FLDS += ["RETURN_COM_EQY","PX_TO_BOOK_RATIO","BEST_PX_BPS_RATIO","BEST_EPS_NUMEST"]



#==============================================================================
# MKT CAP WEIGHTAGE
#==============================================================================

mkt = sids.MKT_CAP_LAST_TRD #halted cut_mkt_cap not working
who = cmgr[mkt.index].NAME
mkt = pd.concat([mkt,who], axis=1, join='inner')
mkt = mkt.set_index("NAME")
mkt = (mkt.apply(lambda z: z/sum(z), axis=0)*100).sort('MKT_CAP_LAST_TRD',ascending=False)
mkt


    

#==============================================================================
# sector EPS
#==============================================================================

filename = 'AxJ ICB Sector Breakdown.xlsm'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

sct='Iron Steel'
df = pd.read_excel(filename,sheetname=sct)
#print the column names
print df.columns

#get a data frame with selected columns
FORMAT = ['Ticker']
CT = df[FORMAT]
for i in CT: print CT[i]

setup_bbg()
start_date = '1/1/2014'
starting = time.time()
sids = cmgr[CT.Ticker.tolist()]
FLDS = ["IS_EPS","BEST_EPS","BEST_EPS_LO","BEST_EPS_HI","CUR_MKT_CAP","BEST_EPS_NUMEST"]

x = sids.get_historical(FLDS, start_date, date.today(), currency="US") #ISO FX
x.backup = x.copy
print(time.time()-starting)

# fill na
x = x.fillna(method="ffill")
#x = x.fillna(method="bfill") # hindsight biases
x.tail()

inspect_flds(x)

# filter est # for reliable sector average
nf=numest_filter(CT.Ticker,'BEST_EPS_NUMEST',5)
meet_criteria=list(itertools.product(nf.index.tolist(), x.columns.levels[1]))
sector_ready = x.loc[:,meet_criteria]
inspect_flds(sector_ready)

# sector mkt cap weighted average 
sector_ready.tail(5)
sector_ready.columns.levels[1].values
for i in _: print i

a = sector_ready.copy()
a = a.drop('CUR_MKT_CAP',axis=1,level=1) # del all same name columns in level 1
a.tail()

inspect.getargspec(plot_flds_counter).args
# invidual counters, can use domestic currency
plot_flds_counter_gridPDF(x=a, normalise= False, secondary = ['BEST_EPS_NUMEST'],
                  has_bbg_name=True,sz=(15,15),NROW=2,NCOL=2,filename="EPS_Individuals "+sct+".pdf",
                  y_label='EPS (US$)',secondary_y_label='Number of estimates')

out = weighted_average(sector_ready) # need US$ standardised
del out['BEST_EPS_NUMEST']
out.plot(title='AxJ ICB Industry ('+sct+') Mkt-Cap Weighted Average')


#==============================================================================
# earn yields
#==============================================================================
filename = 'AxJ ICB Sector Breakdown.xlsm'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Iron Steel')
#print the column names
print df.columns

#get a data frame with selected columns
FORMAT = ['Ticker']
CT = df[FORMAT]
for i in CT: print CT[i]

setup_bbg()
start_date = '1/1/2014'
end_date = date.today()
starting = time.time()
sids = cmgr[CT.Ticker.tolist()]
FLDS = ["EARN_YLD","PX_LAST","CUR_MKT_CAP"]
x = sids.get_historical(FLDS, start_date, end_date, currency="US") #ISO FX
x.backup = x.copy
print(time.time()-starting)

# fill na
x = x.fillna(method="ffill")
#x = x.fillna(method="bfill") # hindsight biases
x.tail()

inspect_flds(x)
print x.shape

a = x.copy()
a = a.drop('CUR_MKT_CAP',axis=1,level=1) # del all same name columns in level 1
a.tail()

inspect.getargspec(plot_flds_counter).args
# invidual counters, can use domestic currency
plot_flds_counter_gridPDF(x=a, normalise= False,secondary=["PX_LAST"],
                  has_bbg_name=True,sz=(10,10),NROW=3,NCOL=3,filename="Earnings_Yield "+sct+".pdf",
                  y_label='Earning Yield (US$ per dollar of stock)',
                  secondary_y_label='Stock Price')


out = weighted_average(x) # need US$ standardised
df_dropna(out).plot(title='AxJ ICB Industry ('+sct+') Mkt-Cap Weighted Average',secondary_y='PX_LAST')


#==============================================================================
# P/B
#==============================================================================
filename = 'AxJ ICB Sector Breakdown.xlsm'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Iron Steel')
#print the column names
print df.columns

#get a data frame with selected columns
FORMAT = ['Ticker']
CT = df[FORMAT]
for i in CT: print CT[i]

setup_bbg()
start_date = '1/1/2014'
end_date = date.today()
starting = time.time()
sids = cmgr[CT.Ticker.tolist()]
FLDS = ["PX_TO_BOOK_RATIO","BEST_PX_BPS_RATIO","CUR_MKT_CAP","PX_LAST"]
x = sids.get_historical(FLDS, start_date, end_date, currency="US") #ISO FX
x.backup = x.copy
print(time.time()-starting)

# fill na
x = x.fillna(method="ffill")
#x = x.fillna(method="bfill") # hindsight biases
x.tail()

inspect_flds(x)
print x.shape

a = x.copy()
a = a.drop('CUR_MKT_CAP',axis=1,level=1) # del all same name columns in level 1
a.tail()

inspect.getargspec(plot_flds_counter).args
# invidual counters, can use domestic currency
plot_flds_counter_gridPDF(x=a, normalise= False,secondary=["PX_LAST"],
                  has_bbg_name=True,sz=(15,10),NROW=2,NCOL=2,
                  filename="price_book "+sct+" "+str(date.today())+".pdf",
                  y_label='P/B',secondary_y_label='Stock Price')


out = weighted_average(x) # need US$ standardised
del out['BEST_PX_BPS_RATIO'] # SOLVE NA MKT CAP WEIGHT LATER
df_dropna(out).plot(title='AxJ ICB Industry ('+sct+') Mkt-Cap Weighted Average',secondary_y='PX_LAST')


#==============================================================================
# RV Profitability Metrics
#==============================================================================
filename = 'AxJ ICB Sector Breakdown.xlsm'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Iron Steel')
#print the column names
print df.columns

#get a data frame with selected columns
FORMAT = ['Ticker']
CT = df[FORMAT]
for i in CT: print CT[i]

setup_bbg()
start_date = '1/1/2014'
end_date = date.today()
starting = time.time()
sids = cmgr[CT.Ticker.tolist()]
FLDS = ["TRAIL_12M_GROSS_MARGIN","OPER_MARGIN","TRAIL_12M_PROF_MARGIN","CUR_MKT_CAP","PX_LAST"]

x = sids.get_historical(FLDS, start_date, end_date, currency="US") #ISO FX
x.backup = x.copy
print(time.time()-starting)

# fill na
x = x.fillna(method="ffill")
#x = x.fillna(method="bfill") # hindsight biases
x.tail()

inspect_flds(x)
print x.shape


a = x.copy()
a = a.drop('CUR_MKT_CAP',axis=1,level=1) # del all same name columns in level 1
a.tail()


inspect.getargspec(plot_flds_counter).args
# invidual counters, can use domestic currency
plot_flds_counter_gridPDF(x=a, normalise= False,secondary=["PX_LAST"],
                  has_bbg_name=True,sz=(15,10),NROW=2,NCOL=2,
                  filename="RV_Profitability "+sct+" "+str(date.today())+".pdf",
                  y_label='P/B',secondary_y_label='Stock Price')


out = weighted_average(x) # need US$ standardised
del out['TRAIL_12M_GROSS_MARGIN']
df_dropna(out).plot(title='AxJ ICB Industry ('+sct+') Mkt-Cap Weighted Average',secondary_y='PX_LAST')


#==============================================================================
# RV ROE
#==============================================================================
filename = 'AxJ ICB Sector Breakdown.xlsm'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Iron Steel')
#print the column names
print df.columns

#get a data frame with selected columns
FORMAT = ['Ticker']
CT = df[FORMAT]
for i in CT: print CT[i]

setup_bbg()
start_date = '1/1/2014'
end_date = date.today()
starting = time.time()
sids = cmgr[CT.Ticker.tolist()]
FLDS = ["RETURN_COM_EQY","BEST_ROE","CUR_MKT_CAP","PX_LAST"]

x = sids.get_historical(FLDS, start_date, end_date, currency="US") #ISO FX
x.backup = x.copy
print(time.time()-starting)

# fill na
x = x.fillna(method="ffill")
#x = x.fillna(method="bfill") # hindsight biases
x.tail()

inspect_flds(x)
print x.shape

a = x.copy()
a = a.drop('CUR_MKT_CAP',axis=1,level=1) # del all same name columns in level 1
a.tail()

inspect.getargspec(plot_flds_counter).args
# invidual counters, can use domestic currency
plot_flds_counter_gridPDF(x=a, normalise= False,secondary=["PX_LAST"],
                  has_bbg_name=True,sz=(15,10),NROW=2,NCOL=2,
                  filename="RV_Profitability "+sct+" "+str(date.today())+".pdf",
                  y_label='P/B',secondary_y_label='Stock Price')

out = weighted_average(x) # need US$ standardised
del out['BEST_ROE']
df_dropna(out).plot(title='AxJ ICB Industry ('+sct+') Mkt-Cap Weighted Average',secondary_y='PX_LAST')


#==============================================================================
# HELPER FUNCTIONS
#==============================================================================
    
def valuation_dashboard(saveas):
    import numpy as plotting
    import matplotlib
    from pylab import *


    
    fig = plt.figure(figsize=(6.5,12))
    fig.subplots_adjust(wspace=0.2,hspace=0.2)
    # gs = gridspec.GridSpec(3, 3)
    
    iplot = 10000
    for i in range(total):
        iplot += 1
        ax = fig.add_subplot(iplot, aspect='auto')


    # plot charts
    ax.plot(x,y,'ko')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    plt.tight_layout()
    
    filename_save = saveas + ".pdf"
    plt.savefig(filename_save,bbox_inches='tight')





import string
list(string.ascii_lowercase)[0:20]

