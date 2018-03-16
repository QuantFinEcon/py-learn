import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
import sys
from datetime import date
import time
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
sns.set(style="white")


tickers = ['GGR SP Equity'] + (['KO'+str(i)+' Comdty' for i in range(1,13,1)])
setup_bbg()
sids = cmgr[tickers]
FLDS = ["PX_LAST"]
start_date = "2007-01-01"
end_date = date.today()

x = sids.get_historical(FLDS, start_date, end_date) 
x = x.fillna(method="ffill")
x.columns = x.columns.droplevel(1)
x.tail()
x.plot(secondary_y=['GGR SP Equity'], figsize=(20,20))

x.ix['2016':].plot(secondary_y=['GGR SP Equity'], figsize=(20,20))



tickers = ['STI Index'] + (['UX'+str(i)+' Index' for i in range(1,10,1)])
setup_bbg()
sids = cmgr[tickers]
FLDS = ["PX_LAST"]
start_date = "2007-01-01"
end_date = date.today()

x = sids.get_historical(FLDS, start_date, end_date) 
x = x.fillna(method="ffill")
x.columns = x.columns.droplevel(1)
x.tail()
x.plot(secondary_y=['STI Index'], figsize=(20,20))

x.ix['2016':].plot(secondary_y=['STI Index'], figsize=(20,20))


tickers = ['KEP SP Equity'] + (['CL'+str(i)+' Comdty' for i in range(1,13,1)])
setup_bbg()
sids = cmgr[tickers]
FLDS = ["PX_LAST"]
start_date = "2007-01-01"
end_date = date.today()

x = sids.get_historical(FLDS, start_date, end_date) 
x = x.fillna(method="ffill")
x.columns = x.columns.droplevel(1)
x.tail()
x.plot(secondary_y=['KEP SP Equity'], figsize=(20,20))

x.ix['2016':].plot(secondary_y=['KEP SP Equity'], figsize=(20,20))
