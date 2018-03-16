import pandas as pd
import tia.bbg.datamgr as dm
import numpy
import sys
from datetime import date

"""
    Parameters
    ----------
    sids: bbg security identifier(s)
    fields: bbg field name(s)
    start: (optional) date, date string , or None. If None, defaults to 1 year ago.
    end: (optional) date, date string, or None. If None, defaults to today.
    period: (optional) periodicity of data [DAILY, WEEKLY, MONTHLY, QUARTERLY, SEMI-ANNUAL, YEARLY]
    ignore_security_error: If True, ignore exceptions caused by invalid sids
    ignore_field_error: If True, ignore exceptions caused by invalid fields
    period_adjustment: (ACTUAL, CALENDAR, FISCAL)
                        Set the frequency and calendar type of the output
    currency: ISO Code
              Amends the value from local to desired currency
    override_option: (OVERRIDE_OPTION_CLOSE | OVERRIDE_OPTION_GPA) DICT KEY VALUE
    pricing_option: (PRICING_OPTION_PRICE | PRICING_OPTION_YIELD)
    non_trading_day_fill_option: (NON_TRADING_WEEKDAYS | ALL_CALENDAR_DAYS | ACTIVE_DAYS_ONLY)
    non_trading_day_fill_method: (PREVIOUS_VALUE | NIL_VALUE)
    calendar_code_override: 2 letter county iso code
    """

# create a DataManager for simpler api access
mgr = dm.BbgDataManager()
# retrieve a single security accessor from the manager
msft = mgr['656 HK Equity']

# Have the manager default to returning a frame instead of values
mgr.sid_result_mode = 'frame'

# Retrieve historical data
msft.get_historical(['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'RETURN_COM_EQY'], 
                    start = '1/1/2014', end = '1/12/2017', period = "DAILY").tail()


# trends in earnings, cashflows, ROEs, leverage, capex and other sector specific leading indicator
FLDS = ["TRAIL_12M_NET_INC_AVAI_COM_SHARE","BEST_NET_INCOME"]

a = ["TRAIL_12M_NET_SALES","BEST_SALES"]
b = ["TRAIL_12M_FREE_CASH_FLOW","BEST_ESTIMATE_FCF"]
c = ["NAV","BEST_NAV"]
d = ["PEGY_RATIO","PE_RATIO"]

x = msft.get_historical(c, '1/1/2010', '1/12/2017')
x = x.fillna(method="ffill")
x = x.fillna(method="bfill")
x.plot()

x.info()



#==============================================================================
# Multiple securities access
#==============================================================================
sids = mgr['MSFT US EQUITY', 'IBM US EQUITY', 'CSCO US EQUITY']
sids.PX_LAST

sids.get_historical('PX_LAST', '1/1/2014', date.today()).tail()

x=sids.get_historical(['PX_OPEN', 'PX_LAST'], '1/1/2014', date.today()).tail()
x
x.info()
x[('IBM US EQUITY', 'PX_LAST')]

#==============================================================================
# caching
#==============================================================================
# ability to cache requests in memory or in h5 file
#
ms = dm.MemoryStorage() #default compression
cmgr = dm.CachedDataManager(mgr, ms, pd.datetime.now())

cmsft = cmgr['MSFT US EQUITY']
cmsft.PX_LAST

%timeit msft.PX_LAST
%timeit cmsft.PX_LAST

csids = cmgr['MSFT US EQUITY', 'IBM US EQUITY']
sids = mgr['MSFT US EQUITY', 'IBM US EQUITY']

%timeit sids.get_historical('PX_LAST', start='1/3/2000', end='1/3/2014').head()
#%timeit csids.get_historical('PX_LAST', start='1/3/2000', end='1/3/2014').head()

#==============================================================================
# # HD Storage - erro feb 17
#==============================================================================
import tempfile
fh, fp = tempfile.mkstemp()

h5storage = dm.HDFStorage(fp)  # Can set compression level for smaller files
h5mgr = dm.CachedDataManager(mgr, h5storage, pd.datetime.now())
h5msft = h5mgr['MSFT US EQUITY']
%timeit h5msft.PX_LAST

h5msft.get_historical('PX_LAST', start='1/2/2000', end='1/2/2014').head()

# notice only IBM gets warning as MSFT is already cached, so it only retrieves IBM data
h5sids = h5mgr['MSFT US EQUITY', 'IBM US EQUITY']
h5sids.get_historical('PX_LAST', start='1/3/2000', end='1/2/2014').tail()



























