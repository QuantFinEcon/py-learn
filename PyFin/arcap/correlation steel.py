import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
import sys
from datetime import date
import seaborn as sns; sns.set(style="white") ;# sns.reset_orig()
import matplotlib.pyplot as plt


filename = 'risk tickers.xlsx'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Iron Steel')
#print the column names
print df.columns

#get the values for a given column
v = df['Tickers'].values
for i in v: print i
v.shape

#get a data frame with selected columns
FORMAT = ['Tickers']
df_selected = df[FORMAT]
df_selected
for i in df_selected: print df_selected[i]

mgr = dm.BbgDataManager()
sids = mgr[df_selected['Tickers'].tolist()]
sids.PX_LAST

x = sids.get_historical('PX_LAST', '1/1/2014', date.today())
x = x.fillna(method="ffill")
#x = x.fillna(method="bfill")
x.tail()

x.plot()

#==============================================================================
# correlation
#==============================================================================

from pydoc import help
from scipy.stats.stats import pearsonr

help(pearsonr)

def clustered_corr(x,sz=(20,20),dropNa=True,filename='testcorr3.pdf'):
    #from scipy.spatial.distance import pdist, squareform
    #from scipy.cluster.hierarchy import linkage, dendrogram
    import scipy.cluster.hierarchy as sch
    import matplotlib.pyplot as plt
    
    if dropNa is True: x=df_dropna(x)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate a mask for the upper triangle
    C = x.corr(method="pearson")
    mask = np.zeros_like(C, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # get bbg names
    sids = mgr[C.columns.values]
    name = sids.NAME
    Y = sch.linkage(C, method='average',metric='correlation') # D is dist matrix
    Z = sch.dendrogram(Y, orientation='right', labels=x.columns.tolist(),no_plot=True) # label before shuffling
    index = Z['leaves']
    C = C.iloc[index,:]
    C = C.iloc[:,index] # clustered dist matrix
    
    pdf_pages = PdfPages(filename)
    fig, ax = plt.subplots(figsize=sz) 
    sns.heatmap(C, cmap=cmap, square = True, annot=True, 
                ax=ax, linewidths=.3, mask=mask,
                annot_kws={'size': 5}, fmt='.2f',
                xticklabels=name['NAME'].values,
                yticklabels=name['NAME'].values)
    
    ax.tick_params(labelsize=5,direction='out')
    plt.tight_layout()
    pdf_pages.savefig(fig)
    pdf_pages.close()
    return C


clustered_corr(x)


sns.pairplot(C)


#==============================================================================
# clustering correlation
#==============================================================================
from scipy.spatial.distance import pdist, squareform

x.shape
y=pdist(x, metric='correlation') #condensed
y = squareform(y)

from scipy.cluster.hierarchy import linkage, dendrogram

z=linkage(x,method='single',metric='correlation')
z.shape
dendrogram(z, color_threshold=0)
z.plot()


def heatmap(x, row_header, column_header, row_method,
            column_method, row_metric, column_metric,
            color_gradient, filename):

x.head(4)

heatmap(x.as_matrix(),x.columns.tolist(),x.columns.tolist(),
        row_method='average',column_method='average',
        row_metric='correlation',column_metric='correlation',
        color_gradient='yellow_black_blue',filename="justin1")







