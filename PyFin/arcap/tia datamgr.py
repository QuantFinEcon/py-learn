import pandas as pd
import tia.bbg.datamgr as dm
import numpy
import sys
from datetime import date

# create a DataManager for simpler api access
mgr = dm.BbgDataManager()
# retrieve a single security accessor from the manager
msft = mgr['MSFT US EQUITY']

#  Can now access any Bloomberg field (as long as it is upper case)
msft.PX_LAST, msft.PX_OPEN

# Access multiple fields at the same time
msft['PX_LAST', 'PX_OPEN']

# OR pass an array
msft[['PX_LAST', 'PX_OPEN']]

# Have the manager default to returning a frame instead of values
mgr.sid_result_mode = 'frame'

# Retrieve historical data
msft.get_historical(['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST'], '1/1/2014', '1/12/2014').head()

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











