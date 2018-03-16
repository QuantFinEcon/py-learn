import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
import sys
from datetime import date
import seaborn as sns; sns.set(style="white")
import matplotlib.pyplot as plt

#==============================================================================
# get FLDS
#==============================================================================

filename = 'risk tickers.xlsx'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Fundamental')
#print the column names
print df.columns

#get the values for a given column
FLDS = df['Metric'].values
for i in FLDS: print i
FLDS.shape
FLDS = FLDS.tolist()
FLDS = [x for x in FLDS if str(x) != 'nan']

#==============================================================================
# get tickers
#==============================================================================

xls = pd.ExcelFile('AxJ ICB Sector Breakdown.xlsm')
sht = xls.sheet_names
for i in sht: print i
df = pd.read_excel('AxJ ICB Sector Breakdown.xlsm',sheetname='Iron Steel')
#print the column names
print df.columns
#get the values for a given column
values = df['Ticker'].values
for i in values: print i

#get a data frame with selected columns
FORMAT = ['Ticker']
df_selected = df[FORMAT]
df_selected
for i in df_selected: print df_selected[i]

mgr = dm.BbgDataManager()
sids = mgr[df_selected['Ticker'].tolist()]
#sids.PX_LAST
FLDS

x = sids.get_historical(["PE_RATIO","GROSS_MARGIN"], '1/1/2014', date.today())

#x = sids.get_historical(FLDS[3:7], '1/1/2014', date.today())
#x = sids.get_historical(FLDS[:25], '1/1/2014', date.today())

x = x.fillna(method="ffill")
x = x.fillna(method="bfill")
x.tail()
x.plot()

idx = pd.IndexSlice
x.loc[:,idx[:,'PE_RATIO']]


x.head(5)
x['5471 JP Equity']['PE_RATIO']
x[]['PE_RATIO']
x[(:,'GROSS_MARGIN')]
list(x)

y = x.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
y['5471 JP Equity'].plot()
y.plot()

from pydoc import help
help(plot)
help(pd.DataFrame.plot)


fig, axes = plt.subplots(nrows=2, ncols=2)

x.plot(figsize=(25,25),subplots=True)

x

x['5471 JP Equity'].plot(figsize=(20,20))
x.tail()

x.plot()

#==============================================================================
# dashboard
#==============================================================================










