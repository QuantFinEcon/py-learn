import quandl
quandl.ApiConfig.api_key = "dDwDihkoAYTL9Tp2rivQ"

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)
    
#==============================================================================
# COT
#==============================================================================

def getCOT(tickers,start,end):
  for ticker in tickers:
    data = quandl.get(ticker,start_date=start, end_date=end)
    print data.tail(1).T
    #data.info()
    #data.tail(1).T
    #data.head(1).T
    print ticker
    lev_pos = data.iloc[:,[7,8]]
    lev_pos.columns = ticker + "  " + lev_pos.columns
    lev_pos.plot()         
    am_pos = data.iloc[:,[4,5]]
    am_pos.columns = ticker + "  " + am_pos.columns
    am_pos.plot()     
         
def getCOT_commod(tickers,start,end):
  for ticker in tickers:
    data = quandl.get(ticker,start_date=start, end_date=end)
    print data.tail(1).T
    #data.info()
    #data.tail(1).T
    #data.head(1).T
    print ticker
    mm = data.iloc[:,[6,7]]
    mm.columns = ticker + "  " + mm.columns
    mm.plot()         
    prod = data.iloc[:,[1,2]]
    prod.columns = ticker + "  " + prod.columns
    prod.plot()
    dealer = data.iloc[:,[3,4]]
    dealer.columns = ticker + "  " + dealer.columns
    dealer.plot()
    


import datetime    
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.today()

tickers=["CFTC/TIFF_CME_JY_ALL",
         "CFTC/TIFF_CME_EC_ALL",
         "CFTC/TIFF_CME_BP_ALL",
         "CFTC/TIFF_CME_SF_ALL",
         "CFTC/TIFF_CME_AD_ALL",
         "CFTC/TIFF_CME_NE_ALL",
         "CFTC/TIFF_CME_CD_ALL",
         "CFTC/TIFF_ICE_DX_ALL"]
getCOT(tickers,start,end)

getCOT(["CFTC/SPC_FO_ALL"])
getCOT(["CFTC/11C_FO_ALL"])
getCOT(["CFTC/SPC_F_ALL"])
getCOT(["CFTC/MFS_FO_ALL"])


getCOT(["CFTC/TIFF_CBOE_VX_ALL"])


#==============================================================================
# GOLD
#==============================================================================
getCOT_commod(["CFTC/GC_FO_ALL"],start,end)
data = quandl.get(["CFTC/GC_FO_ALL"],start_date=start, end_date=end)
data.tail(1).T

tickers = ["WGC/GOLD_DAILY_AUD",
           "WGC/GOLD_DAILY_CAD",
           "WGC/GOLD_DAILY_CHF",
           "WGC/GOLD_DAILY_CNY",
           "WGC/GOLD_DAILY_GBP",
           "WGC/GOLD_DAILY_INR",
           "WGC/GOLD_DAILY_JPY",
           "WGC/GOLD_DAILY_USD",
           ]

data = data = quandl.get(tickers,start_date=start, end_date=end)
data.tail(1).T

data2 = data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)), axis=0)
data2.ix['2016':].plot()

data4 = data.apply(lambda x: 1/x,axis=0)
data3 = data4.ix['2016':].apply(lambda x: (x/x[0]), axis=0)
data3.plot(color=tableau20)

tickers = ["CHRIS/SHFE_AU1","YAHOO/INDEX_HUI","CHRIS/CME_GC1"]
data = data = quandl.get(tickers,start_date=start, end_date=end)
data.tail(1).T
s=data[["CHRIS/SHFE_AU1 - Close","YAHOO/INDEX_HUI - Adjusted Close","CHRIS/CME_GC1 - Last"]]
s.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)), axis=0).ix['2014':].plot(color=tableau20)



         
#==============================================================================
#          
#==============================================================================
         
         

import pandas_datareader.data as web


df = web.DataReader("EURUSD=x", 'yahoo', start, end)


from yahoo_finance import Currency
fx = Currency('EURUSD')
fx.get_bid()
fx.get_ask()
fx.get_rate()
fx.get_trade_datetime()


fx.get_historical('2014-04-25', '2014-04-29')
fx.data_set
