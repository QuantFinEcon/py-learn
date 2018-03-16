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
from sklearn.decomposition import PCA

import seaborn as sns
sns.set(style="white")

#==============================================================================
# small functions, generic, repeatable functions
#==============================================================================
normalise_max_min = lambda x: (x-x.mean())/(x.max()-x.min())
mad = lambda x: np.fabs(x - x.mean()).mean()
sqr = lambda x: np.power(x,2)
#x.apply(mad)
#x.apply(normalise_max_min)
#C.apply(sqr,axis=1).tail()

np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=100)

def pretty_matrix(mat,_min=1e-5,digits=5):
    from scipy.stats import threshold
    return np.round(threshold(mat, _min),digits)

def log_return(x):
    p = np.log(x/x.shift(periods=1))
    p = p.dropna(axis=0,how='any')
    return p

#==============================================================================
# SET UP BBG ENVIRONMENT
#==============================================================================

def setup_bbg():
    # Globals in Python are global to a module, not across all modules
    global mgr
    mgr = dm.BbgDataManager()
    ms = dm.MemoryStorage() #default compression
    # cache for faster retrival
    global cmgr
    cmgr = dm.CachedDataManager(mgr, ms, pd.datetime.now()) # rerun if issue storing data
    print "New bbg cache created!"
                               

def inspect_flds(x):
    x = x.fillna(method="ffill")
    x = x.fillna(method="bfill")
    if x.shape[1]>1: 
        if (type(x.columns) is pd.indexes.multi.MultiIndex):
            # after slicing, levels still remain, but index changes
            # allows linking with old multi index dataframe
            for i in pd.unique(x.columns.levels[0][x.columns.labels[0]]):
                print i
                print x[i].tail(1).T
                print "\n"
        else: 
            print x.tail(1).T
    else: print x


def numest_filter(tickers,filter_fld='BEST_EPS_NUMEST',min_est=5):
    setup_bbg()                           
    # check type of inputs 
    if type(tickers) is not list : tickers = tickers.tolist()
    
    sids = cmgr[tickers]
    NUMEST = sids[filter_fld].sort_values(by=filter_fld,ascending=False) # filter <5 estimates
    #float(NUMEST[NUMEST['BEST_EPS_NUMEST']>5].shape[0])/NUMEST.shape[0]
    print NUMEST
    return NUMEST[NUMEST[filter_fld]>min_est]




def market_cap_share(tickers,start_date,end_date=date.today()):
    setup_bbg()
                               
    # check type of inputs 
    if type(tickers) is not list : tickers = tickers.tolist()
    
    sids = cmgr[tickers]
    FLDS = "CUR_MKT_CAP"
    x = sids.get_historical(FLDS, start_date, end_date, currency="US") #ISO FX
    x = x.fillna(method="ffill")
    x = x.fillna(method="bfill")
    # x.tail()
    # proportion by row
    x = x.apply(lambda z: z/sum(z), axis = 1)
    
    #x.iloc[:,1:4].plot.bar(stacked=True)
    #x.columns = x.columns.droplevel(1)    
    return(x)

def str_index(x,dataname,idx):
    print type(x)
    if isinstance(x, (np.ndarray,np.generic)):
        return dataname+"[:,"+str(idx)+"]"
    elif isinstance(x, pd.DataFrame):
        return dataname+".iloc[:,"+str(idx)+"]"
    else:
        return "unfound datatype.. Update str_index function"

#exec("PP="+str_index(x,"x",[1,2]))

      
def plot_pdf_level_one(data,nrow=1,ncol=1,total=1,sz=(10,8),plot_levels=[],minor_ticks=True,filename="test.pdf"):
    import matplotlib.gridspec as gridspec
    import math
    
    from matplotlib.dates import DateFormatter, MonthLocator
    months = MonthLocator()
    monthsFmt = DateFormatter("%b '%y")

    #A4 ideal figsize landscape
    #optimal=(8.27, 11.69)
    #sz = tuple([optimal[0]/ncol, optimal[1]/nrow])
    
    #local copy of data
    local_copy = data.copy()
    
    pages = int(max(1,math.ceil(float(data.shape[1])/(nrow*ncol))))
    # The PDF document
    pdf_pages = PdfPages(filename)
    base = 0
    #create one pdf page
    for j in xrange(pages):
        plots_per_pg = nrow*ncol
        # Create a figure instance (ie. a new page)
        fig = plt.figure(figsize=sz)
        gs = gridspec.GridSpec(nrow, ncol)
        # plot 
        for i in range(base+0,min(total,base+nrow*ncol),1):            
            # extract data for ndarray, pd.DataFrame
            print i
            exec("a"+str(i)+"="+str_index(local_copy,"local_copy",i))
            exec("ax"+str(i)+"="+"plt.subplot("+"gs["+str(i % plots_per_pg)+"]"+")") # choose loc in grid
            exec("ax"+str(i)+"."+"plot("+"a"+str(i)+")") # plot in grid
            exec("ax"+str(i)+".grid(which='major', linestyle='-', linewidth='0.5', color='#8b8989')") # plot major gridlines
            if minor_ticks == True:
                exec("ax"+str(i)+".minorticks_on()")
                exec("ax"+str(i)+".grid(which='minor', linestyle=':', linewidth='0.3', color='#cdc9c9')") # plot minor gridlines
            if plot_levels != []:
                for lvl in plot_levels: 
                    exec("ax"+str(i)+".axhline(y="+str(lvl)+",linewidth=0.5,color='black')")
            #exec("ax"+str(i)+".set_xticklabels("+"a"+str(i)+".index"+",rotation=90)") # rotation of x-axis
            exec("ax"+str(i)+".xaxis.set_major_locator(months)") # change date xaxis format
            exec("ax"+str(i)+".xaxis.set_major_formatter(monthsFmt)") # change date xaxis format
            exec("labels = "+"ax"+str(i)+".get_xticklabels()")
            exec("plt.setp(labels, rotation=90, fontsize=8)")
            #exec("ax"+str(i)+"."+"title("+"'"+str(local_copy.columns[i])+"')") # plot title
            # plt.setp([ax1],title="")
            if isinstance(local_copy, pd.DataFrame):
                exec("plt.setp("+"ax"+str(i)+","+"title="+"'"+str(local_copy.columns[i])+"')") # plot title
        base += plots_per_pg
        
        fig.autofmt_xdate()
        plt.tight_layout()
        # Done with the page
        pdf_pages.savefig(fig)
     
    # Write the PDF document to the disk
    pdf_pages.close()

#plot_pdf_level_one(x,nrow=2,ncol=2,total=x.shape[1],sz=(10,8),filename="test.pdf")


def plot_pdf_YX(data,dualax_data,nrow=3,ncol=2,total=1,sz=(10,8),plot_levels=[],
                major_ticks = True,minor_ticks=True,filename="test.pdf"):
    import matplotlib.gridspec as gridspec
    import math
    
    from matplotlib.dates import DateFormatter, MonthLocator
    months = MonthLocator()
    monthsFmt = DateFormatter("%b '%y")

    #A4 ideal figsize landscape
    #optimal=(8.27, 11.69)
    #sz = tuple([optimal[0]/ncol, optimal[1]/nrow])
    
    #local copy of data
    local_copy = data.copy()
    dualax_copy = dualax_data.copy()
    pages = int(max(1,math.ceil(float(total*2)/(nrow*ncol))))
    # The PDF document
    pdf_pages = PdfPages(filename)
    base = 0
    k = 0
    #create one pdf page
    for j in xrange(pages):
        plots_per_pg = nrow*ncol
        # Create a figure instance (ie. a new page)
        fig = plt.figure(figsize=sz)
        gs = gridspec.GridSpec(nrow, ncol)

        # i is position of grid spec
        # k is column index of data
        # plot
        for i in range(base+0,min(total*2,base+nrow*ncol),2):       
            # extract data for ndarray, pd.DataFrame
            print i
            exec("a"+str(i)+"="+str_index(local_copy,"local_copy",k))
            exec("ax"+str(i)+"="+"plt.subplot("+"gs["+str(i % plots_per_pg)+"]"+")") # choose loc in grid
            exec("ax"+str(i)+"."+"plot("+"a"+str(i)+",color='#006400')") # plot in grid
            if major_ticks == True:
                exec("ax"+str(i)+".grid(which='major', linestyle='-', linewidth='0.5', color='#8b8989')") # plot major gridlines
            if minor_ticks == True:
                exec("ax"+str(i)+".minorticks_on()")
                exec("ax"+str(i)+".grid(which='minor', linestyle=':', linewidth='0.3', color='#cdc9c9')") # plot minor gridlines
            if plot_levels != []:
                for lvl in plot_levels: 
                    exec("ax"+str(i)+".axhline(y="+str(lvl)+",linewidth=0.5,color='black')")
            #exec("ax"+str(i)+".set_xticklabels("+"a"+str(i)+".index"+",rotation=90)") # rotation of x-axis
            exec("ax"+str(i)+".xaxis.set_major_locator(months)") # change date xaxis format
            exec("ax"+str(i)+".xaxis.set_major_formatter(monthsFmt)") # change date xaxis format
            exec("labels = "+"ax"+str(i)+".get_xticklabels()")
            exec("plt.setp(labels, rotation=90, fontsize=6)")
            #exec("ax"+str(i)+"."+"title("+"'"+str(local_copy.columns[i])+"')") # plot title
            # plt.setp([ax1],title="")
            if isinstance(local_copy, pd.DataFrame):
                exec("plt.setp("+"ax"+str(i)+","+"title="+"'"+str(local_copy.columns[k])+"')") # plot title
                
            #==============================================================================
            #             # dual axis Y  against Underlying Xi
            #==============================================================================
            
            #t = np.arange(0.01, 10.0, 0.01)
            #s1 = np.exp(t)
            #s2 = np.sin(2 * np.pi * t)
            #
            #
            #fig, ax1 = plt.subplots()
            #ax1.plot(t, s1, 'b-')
            #
            #ax2 = ax1.twinx()
            #s2 = np.sin(2 * np.pi * t)
            #ax2.plot(t, s2, 'r.')
            #
            #fig.tight_layout()
            #plt.show()
            
            # ORDER OF COLUMNS FOR DATA and DUAL DATA IS SAME
            print str(i+total*3) + " dual axis"
            exec("BASE"+"="+str_index(dualax_copy,"dualax_copy",0))
            exec("SECOND"+"="+str_index(dualax_copy,"dualax_copy",k+1))
            
            exec("ax"+str(i+total*3)+"="+"plt.subplot("+"gs["+str((i+1)% plots_per_pg)+"]"+")") # choose loc in grid
            exec("ax"+str(i+total*3)+"."+"plot(BASE,color='#0000ff')") # plot in grid
            exec("ax"+str(i+total*3)+".set_ylabel('"+ str(dualax_copy.columns[0]) +"',color='#0000ff')")
            
            # secondary axis
            exec("ax"+str(i+total*6)+"="+"ax"+str(i+total*3)+".twinx()")
            exec("ax"+str(i+total*6)+".plot(SECOND,color='#ff0000')")
            exec("ax"+str(i+total*6)+".set_ylabel('"+ str(dualax_copy.columns[k+1]) +"',color='#ff0000')")
            
#            exec("ax"+str(i+total*3)+".grid(which='major', linestyle='-', linewidth='0.5', color='#8b8989')") # plot major gridlines 
#            exec("ax"+str(i+total*3)+".minorticks_on()")
#            exec("ax"+str(i+total*3)+".grid(which='minor', linestyle=':', linewidth='0.3', color='#cdc9c9')") # plot minor gridlines

            #exec("ax"+str(i+total)+".set_xticklabels("+"a"+str(i+total)+".index"+",rotation=90)") # rotation of x-axis
            exec("ax"+str(i+total*3)+".xaxis.set_major_locator(months)") # change date xaxis format
            exec("ax"+str(i+total*3)+".xaxis.set_major_formatter(monthsFmt)") # change date xaxis format
            exec("labels = "+"ax"+str(i+total*3)+".get_xticklabels()")
            exec("plt.setp(labels, rotation=90, fontsize=6)")
            #exec("ax"+str(i+total)+"."+"title("+"'"+str(local_copy.columns[i])+"')") # plot title
            # plt.setp([ax1],title="")
            # if isinstance(dualax_copy, pd.DataFrame): exec("plt.setp("+"ax"+str(i+total*3)+","+"title="+"'"+str(dualax_copy.columns[0])+"')") # plot title
            
            k += 1
        base += plots_per_pg
        
        fig.autofmt_xdate(rotation=90)
        plt.tight_layout()
        # Done with the page
        pdf_pages.savefig(fig)
     
    # Write the PDF document to the disk
    pdf_pages.close()



def plot_flds_counter(x, normalise=False, secondary=[], norm_fn=normalise_max_min,
                      y_label='LHS scale',secondary_y_label='RHS scale', has_bbg_name=False,
                      figsize=(20,15)):
    ct = pd.unique(x.columns.levels[0][x.columns.labels[0]].values).tolist()
    if has_bbg_name is True:
        SIDS= cmgr[ct]
        ct_names = SIDS.NAME
        
    if secondary != []:
        for i in ct:
            print 'plotting ' + i
            plt.figure(figsize=figsize)
            if normalise:
                ax = x.xs(i, axis=1, level=0).apply(norm_fn).plot(secondary_y=secondary)
                title = i if has_bbg_name==False else ct_names.T[i][0]
                ax.set_title(title)
                ax.set_ylabel(y_label)
                ax.right_ax.set_ylabel(secondary_y_label)
            else: 
                ax = x.xs(i, axis=1, level=0).plot(secondary_y=secondary)
                title = i if has_bbg_name==False else ct_names.T[i][0]
                ax.set_title(title)
                ax.set_ylabel(y_label)
                ax.right_ax.set_ylabel(secondary_y_label)
    else: 
        for i in ct:
            print 'plotting ' + i
            plt.figure(figsize=figsize)
            if normalise: 
                ax = x.xs(i, axis=1, level=0).apply(norm_fn).plot()
                title = i if has_bbg_name==False else ct_names.T[i][0]
                ax.set_title(title)
                ax.set_ylabel(y_label)
            else: 
                ax = x.xs(i, axis=1, level=0).plot()
                title = i if has_bbg_name==False else ct_names.T[i][0]
                ax.set_title(title)
                ax.set_ylabel(y_label)


def nextPage(pdf_pages,fig):
    fig.autofmt_xdate()
    plt.tight_layout()
    # Done with the page
    pdf_pages.savefig(fig) 

def plot_flds_counter_gridPDF(x, normalise=False, secondary=[], norm_fn=normalise_max_min,
                      y_label='LHS scale',secondary_y_label='RHS scale', has_bbg_name=False
                      ,NROW=1,NCOL=1,total=1,sz=(10,8),filename="test.pdf"):
    ct = pd.unique(x.columns.levels[0][x.columns.labels[0]].values).tolist()
    if has_bbg_name is True:
        SIDS= cmgr[ct]
        ct_names = SIDS.NAME

    #plots_per_pg = NROW*NCOL    
    #pages = int(max(1,math.ceil(float(x.shape[1])/(plots_per_pg))))
    # The PDF document
    pdf_pages = PdfPages(filename)

    nr=0;nc=0;
    # Create a figure instance (ie. a new page)
    fig, axes = plt.subplots(nrows=NROW, ncols=NCOL, figsize=sz) # use axes from .plot(ax=axes[i,j])
    # plot 
    if secondary != []:
        for i in ct:
            print 'plotting ' + i
            if nc > (NCOL-1) : nc=0; nr+=1
            if nr>=NROW : 
                nextPage(pdf_pages,fig); nc=0; nr=0
                fig, axes = plt.subplots(nrows=NROW, ncols=NCOL, figsize=sz) # use axes from .plot(ax=axes[i,j])
                
            if normalise:
                ax = x.xs(i, axis=1, level=0).apply(norm_fn).plot(ax=axes[nr,nc],
                         secondary_y=secondary,legend=True)
            else: 
                ax = x.xs(i, axis=1, level=0).plot(ax=axes[nr,nc],
                         secondary_y=secondary,legend=True)

                title = i if has_bbg_name==False else ct_names.T[i][0]
                ax.set_title(title)
                ax.set_ylabel(y_label)
                ax.right_ax.set_ylabel(secondary_y_label)
            nc+=1 
    else: 
        for i in ct:
            print 'plotting ' + i
            if nc > (NCOL-1) : nc=0; nr+=1
            if nr>=NROW : 
                nextPage(pdf_pages,fig); nc=0; nr=0
                fig, axes = plt.subplots(nrows=NROW, ncols=NCOL, figsize=sz) # use axes from .plot(ax=axes[i,j])

            if normalise: 
                ax = x.xs(i, axis=1, level=0).apply(norm_fn).plot(ax=axes[nr,nc],
                         secondary_y=secondary,legend=True)
            else: 
                ax = x.xs(i, axis=1, level=0).plot(ax=axes[nr,nc],
                         secondary_y=secondary,legend=True)
                
                title = i if has_bbg_name==False else ct_names.T[i][0]
                ax.set_title(title)
                ax.set_ylabel(y_label)
            nc+=1 

    # Write the PDF document to the disk
    pdf_pages.close()



def weighted_average(x,flds=[],baseon="CUR_MKT_CAP"):
    ct = pd.unique(x.columns.levels[0][x.columns.labels[0]]).tolist()
    if flds==[]:
        flds = pd.unique(x.columns.levels[1][x.columns.labels[1]]).tolist()
    # create empty dt with same structure
    out = pd.DataFrame(data=0, columns=flds, 
                       index=x[ct[0]].index)
    S=out[baseon]
    del out[baseon]
    for i in ct:
        for j in flds:
            if j is "CUR_MKT_CAP": continue
            temp = x.xs(i,level=0,axis=1)
            out[j] += temp[j]*temp[baseon]
            S += temp[baseon]
            
    out = out.div(S,axis=0)
    del out[baseon]
    return out

def df_dropna(x,axis=0,how='any'):
    X=x.dropna(axis=0,how='any')
    print "removed {0} rows of Nan".format(x.shape[0]-X.shape[0])    
    print "starting date now: {0}".format(X.index[0])
    print "ending date now: {0}".format(X.index[len(X.index)-1])
    return X


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

#clustered_corr(x)



#==============================================================================
# regression
#==============================================================================
def autolabel(rects):
    """
    barplot - Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height*np.sign(rect.get_x()),
                '%.2f' % height, ha='center', va='bottom')

def plot_coeff(coef):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    idx = np.arange(len(coef))
    width = 0.35
    r1 = ax.bar(idx, regr.coef_, width)
    ax.set_xticks(idx)
    ax.set_xticklabels(X.columns,rotation=90)
    plt.axhline(y=0,linewidth=0.5,color='black')
    autolabel(r1)


def pairwise_R2(y,x,start_date='',lookback=62,min_pd=20,sz=(12,10),filename="pairwise_rolling_R2.pdf"):
    x = x.fillna(method="ffill")
    y = y.fillna(method="ffill")
    if start_date != '':
        x=x.ix[start_date:]
        y=y.ix[start_date:]
    if type(y) == pd.core.frame.DataFrame : y=y.iloc[:,0] # series
    R = x.rolling(window=lookback,min_periods=min_pd,center=True)
    C = R.corr(y,pairwise=True)
    R2 = C.apply(sqr,axis=1)

    # fix Nan and Inf
    R2 = R2.replace(np.inf, np.nan)
    R2 = R2.fillna(method='ffill')
    #R2.describe()

    #rank risk drivers to most recent largest contribution
    R2.tail()
    rk = R2.tail(1).T              
    rk = rk.sort_values(rk.columns[0],ascending=False)
    R2 = R2[rk.index] #len(R2.columns)
    x = x[rk.index]
    
    # place y and underlying risk factor side-by-side
    temp1 = pd.concat([y,x],axis=1)
    
    #get bbg names
    ct = temp1.columns.values.tolist()
    SIDS = cmgr[ct]
    ct_names = SIDS.NAME
    ct_names = ct_names.iloc[:,0].tolist()
    R2.columns = ct_names[1:]
    R2.columns = [re.sub(r"'","",i) for i in R2.columns]
    temp1.columns = ct_names
    temp1.columns = [re.sub(r"'","",i) for i in temp1.columns ]
    
    plot_pdf_YX(R2,temp1,nrow=4,ncol=2,total=R2.shape[1],
                       plot_levels=[0,0.5,1],minor_ticks=False,major_ticks=False,sz=sz,filename=filename)


def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)

        
#==============================================================================
# TRIAL CHANGW
#==============================================================================








