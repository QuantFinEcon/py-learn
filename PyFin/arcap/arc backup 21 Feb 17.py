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

#==============================================================================
# small functions, generic, repeatable functions
#==============================================================================
normalise_max_min = lambda x: (x-x.mean())/(x.max()-x.min())


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


def numest_filter(tickers,min_est=5):
    setup_bbg()                           
    # check type of inputs 
    if type(tickers) is not list : tickers = tickers.tolist()
    
    sids = cmgr[tickers]
    NUMEST = sids.BEST_EPS_NUMEST.sort_values(by='BEST_EPS_NUMEST',ascending=False) # filter <5 estimates
    #float(NUMEST[NUMEST['BEST_EPS_NUMEST']>5].shape[0])/NUMEST.shape[0]
    print NUMEST
    return NUMEST[NUMEST['BEST_EPS_NUMEST']>min_est]




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



def plot_pdf(data,nrow=1,ncol=1,total=1,sz=(10,8),filename="test.pdf"):
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
    for i in xrange(pages):
        plots_per_pg = nrow*ncol
        # Create a figure instance (ie. a new page)
        fig = plt.figure(figsize=sz)
        gs = gridspec.GridSpec(nrow, ncol)
        # plot 
        for i in range(base+0,base+nrow*ncol,1):            
            # extract data for ndarray, pd.DataFrame
            exec("a"+str(i)+"="+str_index(local_copy,"local_copy",i)) 
            exec("ax"+str(i)+"="+"plt.subplot("+"gs["+str(i % plots_per_pg)+"]"+")") # choose loc in grid
            exec("ax"+str(i)+"."+"plot("+"a"+str(i)+")") # plot in grid
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


#==============================================================================
# TRIAL CHANGW
#==============================================================================



   


















