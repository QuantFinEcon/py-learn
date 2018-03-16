from fredapi import Fred
fred = Fred(api_key='9e5aa12b5e0081a4af4cf27112ecc00c')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})



#==============================================================================
# 
#==============================================================================
data = fred.get_series('SP500')
data.describe
data.plot()


#==============================================================================
# 
#==============================================================================
_libor = ['USD1MTD156N','GBP1MTD156N','EUR1MTD156N','JPY1MTD156N','CHF1MTD156N']
usd = fred.get_series('USD1MTD156N')
gbp = fred.get_series('GBP1MTD156N')
carry=gbp-usd
carry.describe
carry.plot()
gbpusd = fred.get_series('DEXUSUK')
gbpusd.plot()

f1=pd.concat([gbpusd,carry],axis=1)
f1.columns = ["gbpusd","carry"]
start_date = "2015-01-01"
f1.ix[start_date:].plot(secondary_y=['carry'],legend=True,mark_right=True)


#==============================================================================
# 
#==============================================================================
usd = fred.get_series('USD3MTD156N')
gbp = fred.get_series('GBP3MTD156N')
carry=gbp-usd
carry.describe
carry.plot()
gbpusd = fred.get_series('DEXUSUK')
gbpusd.plot()

f1=pd.concat([gbpusd,carry],axis=1)
f1.columns = ["gbpusd","carry"]
start_date = "2015-01-01"
f1.ix[start_date:].plot(secondary_y=['carry'],legend=True,mark_right=True)


#==============================================================================
# 
#==============================================================================
usd = fred.get_series('USD6MTD156N')
gbp = fred.get_series('GBP6MTD156N')
carry=gbp-usd
carry.describe
carry.plot()
gbpusd = fred.get_series('DEXUSUK')
gbpusd.plot()

f1=pd.concat([gbpusd,carry],axis=1)
f1.columns = ["gbpusd","carry"]
start_date = "2015-01-01"
f1.ix[start_date:].plot(secondary_y=['carry'],legend=True,mark_right=True)

#==============================================================================
# 
#==============================================================================
usd = fred.get_series('USD12MD156N')
gbp = fred.get_series('GBP12MD156N')
carry=gbp-usd
carry.describe
carry.plot()
gbpusd = fred.get_series('DEXUSUK')
gbpusd.plot()

f1=pd.concat([gbpusd,carry],axis=1)
f1.columns = ["gbpusd","carry"]
start_date = "2015-01-01"
f1.ix[start_date:].plot(secondary_y=['carry'],legend=True,mark_right=True)


#==============================================================================
# 
#==============================================================================
eur = fred.get_series(_libor[2])
carry=eur-usd
carry.describe
carry.plot()
eurusd = fred.get_series('DEXUSEU')
eurusd.plot()

f1=pd.concat([eurusd,carry],axis=1)
f1.columns = ["eurusd","carry"]
start_date = "2010-01-01"
f1.ix[start_date:].plot(secondary_y=['carry'],legend=True,mark_right=True)

#==============================================================================
# 
#==============================================================================
DXY=fred.get_series('DTWEXB')
slope=fred.get_series('T10Y2Y')
f1=pd.concat([DXY,-slope],axis=1)
f1.columns = ["DXY","2y-10y"]
start_date = "2007-01-01"
f1.ix[start_date:].plot(secondary_y=['DXY'],legend=True,mark_right=True)

#==============================================================================
# 
#==============================================================================









