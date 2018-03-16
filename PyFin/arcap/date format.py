import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as dates

idx = pd.date_range('2004-05-01', '2011-07-01')
s = pd.Series(np.random.randn(len(idx)), index=idx)
s.plot()

fig, ax = plt.subplots()
ax.plot_date(idx.to_pydatetime(), s, 'v-')
ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1),interval=1))
ax.xaxis.set_minor_formatter(dates.DateFormatter('%d\n%a'))
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid()
ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n%b\n%Y'))
plt.tight_layout()
plt.show()


