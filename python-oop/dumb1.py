#######################################
# explore rpy2
#######################################




# setting temporary PATH variables
import os
#path to your R installation
os.environ['R_HOME'] = "C:/Users/1580873/Documents/R/R-3.5.0"
os.environ['PATH'] += os.pathsep + "C:/Users/1580873/Documents/R/R-3.5.0/bin"
os.environ['R_USER'] = 'C:/ProgramData/Anaconda3/Lib/site-packages/rpy2'



import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

importr('base')
importr('stats')



# test : evaluating R code
r(
"""
setwd("C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow")
getwd()
"""
)


r(
"""
getwd()
"""
)


r(
'''
# create a function `f`
f <- function(r, verbose=FALSE) {
    if (verbose) {
        cat("I am calling f().\n")
    }
    2 * pi * r
}
# call the function `f` with argument value 3
x = f(3)
'''
)

r.x


namespace=r(
"""
setwd("C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow")
print(getwd())
source("KmeansOD.R")
lsf.str(.GlobalEnv)
"""
)

namespace?
for i in namespace: print(i)


r(
"""
r_namespace <- lsf.str(.GlobalEnv)
"""
)

r.r_namespace


r(
"""
setwd("C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow")
print(getwd())
source("KmeansOD.R")
namespace = lsf.str(.GlobalEnv)

metadata = read.csv(file = 'consolidate_metadata.csv')
head(metadata)
"""
)


list(robjects.globalenv.keys())
list(robjects.baseenv.keys())

'aggregate' in list(robjects.baseenv.keys())
importr('stats')


list(r.metadata.names)
r.metadata.rx2("filename")

pandas2ri.activate()

metadata = r.metadata
metadata.dtypes

pandas2ri.py2ri(metadata)

r("""
print({k})
for(i in 1:10){
    print(i)}  

""".format(k=5))

from string import Template
class Template(Template):
    ''' override string.Template defaults '''
    delimiter = '@'
pandas2ri.activate()

r(
Template("""
for( file in unique(metadata$FilteredFilename)){
  data = subset(metadata[which(metadata$FilteredFilename==file),],
                select = c('size', 'numberofrecords'))
    assign(file, kmod(data, k=@k, l=@l, i_max=@i_max))
    print(paste0(file," Done!"))
}
""").substitute(k=2,l=1,i_max=100)
)


r(
Template("""
data = subset(metadata[which(metadata$FilteredFilename=='SftTradeLevel'),],
                       select = c('size', 'numberofrecords'))
X = kmod(data, k=@k, l=@l, i_max=@i_max)
""").substitute(k=2,l=1,i_max=100)
)

r("data = subset(metadata[which(metadata$FilteredFilename=='SftTradeLevel'),],select = c('size', 'numberofrecords'))")
r.data
r("result = kmod(data, k=2, l=1, i_max=100)")
list(r.result)
result=r("kmod(data, k=2, l=1, i_max=100)")
for i in result: print(i)
list(result.names)
result.rx2('XC_dist_sqr_assign')

for var in list(result.names):
    print(var)
    print(type(result.rx2(var)))
#    print(result.rx2(var))
#    print(var+"=result.rx2('"+var+"')")
    exec(var+"=result.rx2('"+var+"')")
    print(pandas2ri.ri2py_vector(eval(var)))

type(k)
l
C
within_ss
L_dist_sqr

#import pandas.rpy.common as com

pandas2ri.ri2py_dataframe(C)
pandas2ri.ri2py_vector(C)
pandas2ri.ri2py_vector(k)
pandas2ri.ri2py_vector(XC_dist_sqr_assign)
type(_)
dir(pandas2ri)


base = importr('base')
base.summary(data)

from rpy2.robjects import pandas2ri
r_dataframe = pandas2ri.py2ri(data[['size','numberofrecords']])



#r.X = r_dataframe 
list(robjects.globalenv.keys())
robjects.globalenv['X'] = r_dataframe
r("result = kmod(X, k=2, l=1, i_max=100)")
r.kmod(data[['size','numberofrecords']], k=2, l=1, i_max=100)


        """ Run model in R script on data """
        cmd = Template("kmod(X=@data, k=@k, l=@l, i_max=@i_max)")\
                .substitute(data="self.data", 
                            k=self.K, 
                            l=self.L,
                            i_max=self.iterations)
        # execute R scipt in python with rpy2
        r(cmd)
        
        #function (X, k = 5, l = 0, i_max = 100, conv_method = "delta_C", 
        #         conv_error = 0, allow_empty_c = FALSE)
        r.kmod(X=self.data, k=self.K, l=self.L, i_max = self.iterations)



Template(
"""     assign(file, kmod(data, k=$k, l=$l) 

""").substitute(k=2,l=1)




base = importr('base')
r(
"""
lsf.str(package:base)
"""
)

# http://nullege.com/codes/search/rpy2.robjects.vectors.DataFrame









#==============================================================================
# 
#==============================================================================

# import and initialize
import rpy2.rinterface as ri
ri.initr()

# make a function to rename column i
def rename_col_i(m, i, name):
    m.do_slot("dimnames")[1][i] = name

# create a matrix
matrix = ri.baseenv["matrix"]
rlist = ri.baseenv["list"]
m = matrix(ri.baseenv[":"](1, 10),
           nrow = 2,
           dimnames = rlist(ri.StrSexpVector(("1", "2")),
                            ri.StrSexpVector(("a", "b", "c", "d", "e"))))

dir(r)
r.kmod

#######################################
# base
#######################################

#==============================================================================
# Provide an interface for DQ Models
#==============================================================================

import abc

class Model(metaclass=abc.ABCMeta):
    """
    Declare an interface for DQ Models.
    @ data:     Input data in pandas.DataFrame.
                Preprocess data before loading into model.
    @config:    Pass from external YAML/JSON. 
                dict(model's parameter name = value)
    """
    @abc.abstractmethod
    def __init__(self, data=None, config=None):
        """ Be explicit in new instances for input: data, config"""
        pass

    @abc.abstractmethod
    def Execute(self):
        """ Run R/python model with parameters from config YAML """
        pass

    @abc.abstractmethod
    def GetOutliers(self):
        """ Output result of model's execution.
            Standardise the output to show outliers of data. """    
        pass

#######################################
# models
#######################################

from base import Model
from measures import min_max_percentile

import os
from string import Template
class Template(Template): delimiter = '@'

import matplotlib.pyplot as plt

#==============================================================================
# Test
#==============================================================================

#os.chdir("C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow")
#
#config=dict(
#R_HOME = "C:/Users/1580873/Documents/R/R-3.5.0",
#R_USER = 'C:/ProgramData/Anaconda3/Lib/site-packages/rpy2',
#ModelPath = "C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow",
#X = ['size','numberofrecords'],
#K = 2,
#L = 1,
#iterations = 100)
#
#import pandas as pd
#data = pd.read_csv('consolidate_metadata.csv')
#data.dtypes
#data.groupby('FilteredFilename').count()
#data = data.loc[data['FilteredFilename']=='SentryCollateral',:]
#
#
#model=KLmeans(data=data, config=config)
#model.r
#model.Execute()
#model.model_results
#model.GetOutliers()


#==============================================================================
# Instances of DQ Model
#==============================================================================

class KLmeans(Model):
    """
    Objective:  Detect outlier after specifying K clusters
                and L outliers. Calls Rscript KmeansOD.R
    @data:      columnar table
    @config:    dict like YAML/JSON file for passing parameter values
                - X: names of columns to be used in model
                - R_HOME: filepath to R.exe location
                - R_USER: filepath to rpy2
                - ModelPath: filepath of KmeansOD.R script
                - K: forced dataset to K count of clusters
                - L: number of outliers. KLmeans identifies as data points
                     that causes the largest fall in within cluster MSE
                - iterations: stops clustering at end of iterations 
                              or convergence in delta SSE
    """

    r = None
    pandas2ri = None
    
    def __init__(self, data, config):
        self.data = data
        self.predictors = self.response = None
        if 'Y' in config.keys(): self.response = config['Y']
        if 'X' in config.keys(): self.predictors = list(config['X'])
        self.K = config['K']
        self.L = config['L']
        self.iterations = config['iterations']
        self.model_results = dict()

        # set up R env
        os.environ['R_HOME'] = config['R_HOME']
        os.environ['PATH'] += os.pathsep + config['R_HOME'] + "/bin"
        os.environ['R_USER'] = config['R_USER']
        from rpy2.robjects import r, pandas2ri
        KLmeans.r = r
        KLmeans.pandas2ri = pandas2ri        
        KLmeans.pandas2ri.activate()

        # load functions
        KLmeans.r(
        """
        setwd("{0}")
        source("KmeansOD.R")
        """.format(config['ModelPath'])
        )
        pass
    
    def Execute(self):
        """ Run model with KmeansOD.R functions on python's data """
        #function (X, k = 5, l = 0, i_max = 100, conv_method = "delta_C", 
        #         conv_error = 0, allow_empty_c = FALSE)
        if not self.predictors is None:
            _data = self.data[ self.predictors ]
        else:
            _data = self.data
        X = KLmeans.r.kmod(X=_data, k=self.K, l=self.L, i_max = self.iterations)
        
        # convert R list vector to python dictionary        
        for var in list(X.names):
#            print(var)
#            print("self.model_results['"+var+"'] = KLmeans.pandas2ri\
#                  .ri2py_vector(X.rx2('"+var+"'))")
            exec("self.model_results['"+var+"'] = KLmeans.pandas2ri.\
                 ri2py_vector(X.rx2('"+var+"'))")
#            print(self.model_results[var])
            

    def GetOutliers(self):
        """
        Returns: pandas.DataFrame. Rows are records of identified outliers. 
        """
        self.addQuantiles()
        X = self.data.iloc[self.model_results['L_index']-1,:]
        X.loc[:,'SSE'] = self.model_results['L_dist_sqr']
        return X
    
    def addQuantiles(self):
        if not self.predictors is None:
            _predictors = self.predictors
        else:
            _predictors = list(self.data.columns)        
        for predictor in _predictors:
            self.data = min_max_percentile(self.data, predictor)
        pass


#==============================================================================
# Test
#==============================================================================

#config=dict(eps=0.3,metric='euclidean',min_samples=3,
#            X = ['size','numberofrecords'])
#
#import pandas as pd
#data = pd.read_csv('consolidate_metadata.csv')
#data.dtypes
#data.groupby('FilteredFilename').count()
#data = data.loc[data['FilteredFilename']=='SentryCollateral',:]
#
#
#model=DBSCAN(data=data, config=config)
#model.Execute()
#model.model_results
#model.GetOutliers()


#==============================================================================
# Instances of DQ Model
#==============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as _DBSCAN
import numpy as np


class DBSCAN(Model):
    """
    Objective:  Detect outlier with Density-Based Spatial Clustering of 
                Applications with Noise (DBSCAN). 
                Clusters define themselves based on their distance proximity. 
                Noise are given -1 labels. 
    @config:    dict like YAML/JSON file for passing parameter values
                - X: names of columns to be used in model
                - min_samples:  number of points within eps distance 
                                of specified distance metric to be considered 
                                a cluster itself. At least 2 to form distance
                                matrix. 
                - eps: in specified distance metric
                - metric:   measurement of distance between 2 points to be 
                            considered in same cluster. Look at 
                            sklearn.metrics.pairwise.calculate_distance
                            Default: 'euclidean'
    """

    def __init__(self, data, config):
        self.data = data
        self.predictors = self.response = None
        if 'Y' in config.keys(): self.response = config['Y']
        if 'X' in config.keys(): self.predictors = list(config['X'])
        self.eps = config['eps']
        self.min_samples = config['min_samples']
        self.metric = config['metric']
        self.model_results = dict()
        pass
        
    def Execute(self):
        """ Run sklearn.cluster.DBSCAN model """
        if not self.predictors is None:
            _data = self.data[ self.predictors ]
        else:
            _data = self.data
        X = StandardScaler().fit_transform(_data)
        db = _DBSCAN(eps=self.eps, 
                     min_samples=self.min_samples, 
                     metric=self.metric, n_jobs=-1).fit(X)
        labels = db.labels_
        self.model_results['labels'] = labels
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        pass

    def GetOutliers(self):
        """
        Returns: pandas.DataFrame. Rows are records of identified outliers. 
        """
        self.addQuantiles()
        if -1 in set(self.model_results['labels']):
            # -1 is DBSCAN outlier label
            outlier_index = np.where(self.model_results['labels']==-1)[0]
            X = self.data.iloc[outlier_index,:]
            return X
        else:
            print("No outliers to report!")
            return

    def addQuantiles(self):
        if not self.predictors is None:
            _predictors = self.predictors
        else:
            _predictors = list(self.data.columns)        
        for predictor in _predictors:
            self.data = min_max_percentile(self.data, predictor)
        pass

    def PlotOutliers(self):
        outliers = self.GetOutliers()
        self.data['color'] = [0 if i not in list(outliers.index) 
                                else 1 for i in range(self.data.shape[0])]

        fig=plt.figure()
        ax = fig.add_subplot(111)
        X = self.data['size']
        Y = self.data['numberofrecords']
        ax.scatter(X, Y, marker="s", c=self.data['color'])
        #ax.set_title('EBBS')
        ax.set_xlabel('size')
        ax.set_ylabel('number of records')
        plt.show()
        
        
        pass

#######################################
# factory
#######################################

#================================================
# Run DQ Model
#================================================

class BuildModel(object):
    """
    Objective:  Builder class runs any DQ model instance
                build on *Model* interface. 
    """
    def __init__(self, model, data, config):
        self.model = model(data=data, config=config)
        self.config = config
        pass
    
    def get_params(self):
        print(self.model.__doc__)
        print("####################################")
        print("# Config Parameters Values")
        print("####################################")
        for parameter, value in self.config.items():
            print(parameter + " : " + str(value))
        pass

    def execute(self):
        self.model.Execute()
        pass
    
    def getOutliers(self):
        #model.model_results
        return self.model.GetOutliers()
        pass

#######################################
# main
#######################################

import os
os.chdir("C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow")

from datetime import datetime


from proj_DQWorkflow.models import DBSCAN, KLmeans
from proj_DQWorkflow.factory import BuildModel
import proj_pyexcel

proj_pyexcel.editor?


dest_path = "DQreport.xlsx"
writer = pd.ExcelWriter(dest_path)

config=dict(
R_HOME = "C:/Users/1580873/Documents/R/R-3.5.0",
R_USER = 'C:/ProgramData/Anaconda3/Lib/site-packages/rpy2',
ModelPath = "C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow",
X = ['size','numberofrecords'],
K = 2,
L = 3,
iterations = 100)




data = x.loc[x['FilteredFilename']=='CounterpartyPFE.txt',:]

m=BuildModel(model=KLmeans, data=data, config=config)

m.get_params()
m.execute()
outliers = m.getOutliers()

outliers.to_excel(excel_writer = writer,
                  sheet_name = 'KLmeans')

config=dict(eps=0.3,metric='euclidean',min_samples=3,
            X = ['size','numberofrecords'])

m=BuildModel(model=DBSCAN, data=data, config=config)

m.get_params()
m.execute()
outliers = m.getOutliers()

outliers.to_excel(excel_writer = writer,
                  sheet_name = 'DBSCAN')

writer.save()
writer.close()





#==============================================================================
# 
#==============================================================================

filepath = "PIMX.AU.IMEX.BAL.DA.SND00.csv"
FilterFileName(filepath)

history_path = 'C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow/history_path - Copy.txt'
today_path = 'C:/Users/1580873/Desktop/PyProject/proj_DQWorkflow/today_path.txt'
prev_date = "20180515"

saveMetadata(today_path = today_path,
             history_path = history_path)

compareMetadata(today_path, history_path, prev_date, lookback=30)

#==============================================================================
# 
#==============================================================================

def FilterFileName(filepath):
    # remove extension
    filepath = "".join(filepath.split(".")[:-1])    
    # remove char in str conditions
    _remove = lambda x: (x.isdigit() or 
                         x in [".","-","_"])
    #"_D_","_M_","_Q_","_W_"
    #"ALL_CRISKREPMFU_"
    filepath = "".join([w for w in filepath if not _remove(w)])
    return filepath

def compareMetadata(today_path, history_path, prev_date, lookback=30):
    """ 
    Responsibility:
        - compare today's vs yesterday's metadata for each file 
            to derive % change of filesize and line counts
        - benchmark against its own history to derive percentile
        - run builder factory to build model and run for each file
    
    @today_path: today's .txt file
    @history_path: cache of all historical metadata .txt file
    @prev_date: str, YYYYMMDD filter history for this date
    @lookback: recency to percentile
    """
    
    names = ['FilteredFileName','filename','numberofrecords','size','date']
    now = pd.read_csv(today_path, sep="|", header=None, names=names)

    names = ['FilteredFileName','filename','numberofrecords','size','date']
    history = pd.read_csv(history_path, sep="|", header=None, names=names)
    
    history = history.loc[~history['date'].isnull(),:]

    now['date'] = now['date'].astype(int).astype(str)
    history['date'] = history['date'].astype(int).astype(str)

#    # extract FilteredFileName from filename for today metadata
#    now['FilteredFileName'] = None
#    for i in range(now.shape[0]):
#        line = now.iloc[i,0]
#        line = line.split("\\")
#        line = line[-1]
#        "".join(line.split(".")[:-1])
#        now.loc[i,'FilteredFileName'] = FilterFileName(line)

    now.columns = list(now.columns[0:2]) + ["Today_"+x for x in now.columns[2:]]
    
    # filter for prev_date from history
    prev = history.loc[history['date'] == prev_date,:]
    prev.columns = list(prev.columns[0:2]) + ["Yesterday_"+x for x in prev.columns[2:]]
    
    # calculate % change in size and numberofrecords against prev day
    x = pd.merge(now, prev, on='FilteredFileName')
    x['Filesize Change(%)'] = ( x['Today_size'] - x['Yesterday_size'] ) / x['Yesterday_size']
    x['Lines Count Change'] = ( x['Today_numberofrecords'] - x['Yesterday_numberofrecords'] )
    x['Lines Count Change(%)'] = ( x['Today_numberofrecords'] - x['Yesterday_numberofrecords'] ) \
                                    / x['Yesterday_numberofrecords']
    
    # calculate percentile against k period lookback for number range
    x['Lines Count Percentile'] = None
    x['Filesize Percentile'] = None
    x['Model:KLmeans'] = None
    x['Model:DBSCAN'] = None
    
    # extract lookback data from history
    for file in x['FilteredFileName'].unique():
        history_ = history.loc[history['FilteredFileName']==file,:]
        hist_date = sorted(list(history_['date'].unique()),reverse=True)
        lookback_ = min(len(hist_date), lookback)
        hist_date = hist_date[:lookback_]
        history_ = history_.loc[history_['date'].isin(hist_date),:]

        history_size = history_['size']
        _max = max(history_size); _min = min(history_size)
        x.loc[x['FilteredFileName']==file,'Filesize Percentile'] = \
            (x.loc[x['FilteredFileName']==file, 'Today_size'] - _min) / (_max - _min)

        history_numberofrecords = history_['numberofrecords']
        _max = max(history_numberofrecords); _min = min(history_numberofrecords)
        x.loc[x['FilteredFileName']==file,'Lines Count Percentile'] = \
            (x.loc[x['FilteredFileName']==file, 'Today_numberofrecords'] - _min) / (_max - _min)
    
    # run model
    x = x[['FilteredFileName','Filesize Change(%)','Lines Count Change','Lines Count Change(%)',
           'Filesize Percentile', 'Lines Count Percentile']]

    # index as time of DQ check
    x['Time of DQ Check'] = str(datetime.now().strftime('%d-%b-%Y %H:%M:%S'))
    x = x.set_index('Time of DQ Check')
    return x

def saveMetadata(today_path, history_path):
    """
    Responsibility:
        - save metadata from today_path txt to history_path txt
#        - delete today_path txt 
        - write/append results to DQreport.xlsx for display
    Destination:
        - history cache txt: if no outlier detected for today, otherwise, 
                             save to cache of bad metadata. 
        - DQreport.xlsx:     Will be left open and visible for live update of
                             DQ check results. 
    Txt columns:
        filename | FilteredFileName | date | size | numberofrecords
    """
    
    # copy from temporary text to history cache text
    with open(today_path, 'r') as today:
        with open(history_path, 'a') as history:
            for line in today:
                history.write(line)
    
#    os.remove(today_path)

#######################################
# measures
#######################################

#==============================================================================
# Measurement calculations
#==============================================================================

def min_max_percentile(df, column):
    """ 
    Apply columnwise minmax range on a dataframe column 
    @df: pandas dataframe
    @column: column name of dataframe to apply minmax percentile
    """
    _min = df[column].min()
    _max = df[column].max()
    df[column+'_percentile'] = df.apply(func=lambda x: (x[column] - 
          _min) / (_max - _min), axis=1)
    return df


#######################################
# compress
#######################################

import os
import zipfile
import subprocess
from functools import wraps
import time

#import py7zlib 
#import gzip
#import pylzma


def timeit(method):
    """
    method decorator to time completion of a function
    """
    @wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print( '%r  %2.2f ms' %  (method.__name__, (te - ts) * 1000))
        return result
    return timed


@timeit
def zipper(file, remove=True):
    """
    @filepath: "C:\\....\\test.csv"
    """
    ext_len = len(file.rsplit(".")[-1])
    filename = file[:(len(file)-ext_len-1)]
    print(filename)
    #compress file
    zipf = zipfile.ZipFile(filename+'.zip', 'w')
    zipf.write(filename=file,
               arcname=os.path.basename(file),
               compress_type=zipfile.ZIP_DEFLATED)
    zipf.close()
    
    if remove:
        #delete file after zip
        try:
            os.remove(file)
        except OSError:
            print("Error deleting "+file)

@timeit
def zip_all(rootdir, remove=True):
    """
    loop through root dir and for all files in dir, zip each file
    avoid zipping dir
    """
    rootdir = rootdir.rstrip(os.sep)
    for path, dirs, files in os.walk(rootdir):
        # path is str, dirs [of folders to walk in], 
        # files [not folders to save as values]
        print(path, dirs, files)
        print('\n')
        
        # skip dirs
        for file in files:
            ext = file.rsplit(".")[-1]
            # zip if not compressed
            if not ext in ['zip','7z']:
                filepath = path + os.sep + file
                zipper(filepath, remove=remove)

@timeit
def unzipper(file, remove=True):
    """
    @filepath: "C:\\....\\test.zip"
    """
    #compress file
    zipf = zipfile.ZipFile(file, 'r')
    path = file[:file.rfind(os.sep)]
    zipf.extractall(path)
    zipf.close()
    
    if remove:
        #delete file after zip
        try:
            os.remove(file)
        except OSError:
            print("Error deleting "+file)

@timeit
def un7zip(Extract_File, 
           remove=True,
           Z_Location = 'C:\\Program Files\\7-Zip\\7z.exe'):
    """
    @Extract_File: "C:\\....\\test.7z"
    """
    filepath = Extract_File
    Extact_Folder = Extract_File[:Extract_File.rfind(os.path.sep)]
    # -y replace all auto YES
    Extract_Target = r'"{}" e "{}" -o"{}" -y'.format(Z_Location, Extract_File , Extact_Folder)
    print(Extract_Target )
    p=subprocess.Popen(Extract_Target,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell=True)
    while p.poll() is None:
        pass
    p.terminate()
    
    print(filepath)
    if remove:
        #delete file after zip
        try:
            os.remove(filepath)
        except OSError:
            print("Error deleting " + filepath)


@timeit
def unzip_all(rootdir, remove=True):
    """
    loop through root dir and for all files in dir, zip each file
    avoid zipping dir
    """
    rootdir = rootdir.rstrip(os.sep)
    for path, dirs, files in os.walk(rootdir):
        # path is str, dirs [of folders to walk in], 
        # files [not folders to save as values]
        print(path, dirs, files)
        print('\n')
        
        # skip dirs
        for file in files:
            ext = file.rsplit(".")[-1]
            filepath = path + os.sep + file
            print(filepath)
            
            try:
                # zip if not compressed
                if ext in ['zip']:
                    unzipper(filepath)
                elif ext in ['7z']:
                    un7zip(filepath,remove=remove)
                else:
                    print("Unsupported file extension: " + ext)
            except:
                error.append(filepath)
            
            print('\n')



#==============================================================================
# https://www.dotnetperls.com/7-zip-examples
#==============================================================================

rootdir = r'C:\MFU\Analyse Metadata\Unextracted'

#rootdir = r'C:\\MFU\\Analyse Metadata\\Not Extracted 3\\201710\\20171030'

#rootdir='C:\\MFU\\Analyse Metadata\\Extracted\\201703\\20170309-incomplete'

error=[]

unzip_all(rootdir,remove=True)

zip_all(rootdir)

#####################################
# extract metadata
#####################################

def extractMetadata(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    data = pd.DataFrame(columns=['FilteredFilename','filename','numberofrecords','size','cob'])
    rootdir = rootdir.rstrip(os.sep)
    for path, dirs, files in os.walk(rootdir):
        # path is str, dirs [of folders to walk in], files [not folders to save as values]
        print(path, dirs, files)
        print('\n')

        for file in files: 
            filepath = path + os.sep + file
            print(filepath)
            extension = file.split(".")[-1]
            if extension.lower() in ['csv','txt']:
                length = get_flatfile_length(filepath)
                filesize = os.path.getsize(filepath)/1000 #kbytes 
                filter_filename = FilterFileName(file)

                cob = re.findall(r'(\d{8})', path)
                if len(cob) == 0:
                    cob = re.findall(r'(\d{8})', file)
                if len(cob) > 0:
                    cob = cob[0]
                else:
                    cob = ""
                
                filename = os.path.basename(filepath)
                data.loc[len(data)] = [filter_filename,filename,
                                         length,filesize,
                                         cob]
            elif extension in ['7z','zip']:
                pass
            else:
                print("Unsupported file extension!")
    return data

##############################
# H2O presentation
##############################




import h2o
# open FLOW at http://localhost:54321/flow/index.html
h2o.init(ip="127.0.0.1", 
         port=54321,
         nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "4G")  #max mem size is the maximum memory to allocate to H2O)

metadata_path = "C:/Users/1580873/Desktop/Completed_Developments/H2O/consolidate_metadata.csv"
iris_path = 'C:/Users/1580873/Desktop/Completed_Developments/H2O/iris.csv'

#==============================================================================
# multi class logistic
#==============================================================================

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

h2o_df = h2o.import_file(path = metadata_path,
                         parse=True,header=0,sep=",")

h2o_df['FilteredFilename'] = h2o_df['FilteredFilename'].asfactor()
h2o_df.types
h2o_df.columns

train, test = h2o_df.split_frame(ratios=[0.90], seed=1) # split into 0.9, 0.1
train.shape
test.shape

multinomial_glm = H2OGeneralizedLinearEstimator(family="multinomial")
multinomial_glm.train(x=["size","numberofrecords"], 
                   y='FilteredFilename',
                   training_frame=train)

multinomial_glm.show()
multinomial_glm.get_params().keys()

#import yaml
#with open('test.yml','w') as f:
#    yaml.dump(multinomial_glm.get_params(), stream=f, indent=4)

predictions = multinomial_glm.predict(test)
predictions.show()


model = multinomial_glm
dir(model)

print('Model Type:', model.type)
print('logloss', model.logloss(valid = False))
print('R2', model.r2(valid = False))
print('RMSE', model.rmse(valid = False))

#print('Accuracy', model.accuracy(valid = False))
#print('AUC', model.auc(valid = False))
#print('Error', model.error(valid = False))
#print('MCC', model.mcc(valid = False))


#==============================================================================
# deep learning autoencoder for anomaly detection
#==============================================================================

h2o_df = h2o.import_file(path = metadata_path,
                         parse=True,header=0,sep=",")

h2o_df['FilteredFilename'] = h2o_df['FilteredFilename'].asfactor()
h2o_df.as_data_frame().head(5)
h2o_df.types
h2o_df.columns
#h2o_df = h2o_df[['FilteredFilename','size','numberofrecords']]

train, test = h2o_df.split_frame(ratios=[0.7])
train.shape
test.shape

# Train deep autoencoder learning model on "normal"
# training data, y ignored

#dir(h2o.estimators.deeplearning)
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

anomaly_model = H2OAutoEncoderEstimator(activation = "Tanh",
                                        hidden = [50,50,50],
                                        sparse=True,
                                        l1=1e-4,
                                        epochs=100)

anomaly_model.train(y='FilteredFilename', x=['size','numberofrecords'], 
                    training_frame=train)

anomaly_model.get_params().keys()
"""
anomaly_model.get_params().keys()
Out[196]: dict_keys(['model_id', 'training_frame', 'validation_frame', 'nfolds', 'keep_cross_validation_predictions', 'keep_cross_validation_fold_assignment', 'fold_assignment', 'fold_column', 'response_column', 'ignored_columns', 'ignore_const_cols', 'score_each_iteration', 'weights_column', 'offset_column', 'balance_classes', 'class_sampling_factors', 'max_after_balance_size', 'max_confusion_matrix_size', 'max_hit_ratio_k', 'checkpoint', 'pretrained_autoencoder', 'overwrite_with_best_model', 'use_all_factor_levels', 'standardize', 'activation', 'hidden', 'epochs', 'train_samples_per_iteration', 'target_ratio_comm_to_comp', 'seed', 'adaptive_rate', 'rho', 'epsilon', 'rate', 'rate_annealing', 'rate_decay', 'momentum_start', 'momentum_ramp', 'momentum_stable', 'nesterov_accelerated_gradient', 'input_dropout_ratio', 'hidden_dropout_ratios', 'l1', 'l2', 'max_w2', 'initial_weight_distribution', 'initial_weight_scale', 'initial_weights', 'initial_biases', 'loss', 'distribution', 'quantile_alpha', 'tweedie_power', 'huber_alpha', 'score_interval', 'score_training_samples', 'score_validation_samples', 'score_duty_cycle', 'classification_stop', 'regression_stop', 'stopping_rounds', 'stopping_metric', 'stopping_tolerance', 'max_runtime_secs', 'score_validation_sampling', 'diagnostics', 'fast_mode', 'force_load_balance', 'variable_importances', 'replicate_training_data', 'single_node_mode', 'shuffle_training_data', 'missing_values_handling', 'quiet_mode', 'autoencoder', 'sparse', 'col_major', 'average_activation', 'sparsity_beta', 'max_categorical_features', 'reproducible', 'export_weights_and_biases', 'mini_batch_size', 'categorical_encoding', 'elastic_averaging', 'elastic_averaging_moving_rate', 'elastic_averaging_regularization'])
"""


recon_error = anomaly_model.anomaly(test)
recon_error.shape

error = recon_error.as_data_frame()
error.sort_values('Reconstruction.MSE',ascending=False).head(5)
i = list(_.index)

test.as_data_frame().iloc[i,:]

# reconstructured predictions far away from original ==> outliers?
test.as_data_frame().loc[(recon_error.as_data_frame()>0.099),:]


# Note: Testing = Reconstructing the test dataset
test_recon = anomaly_model.predict(test)
predict=test_recon.as_data_frame()
list(predict.columns.values)


#==============================================================================
# integrate with scikit-learn http://scikit-learn.org/stable/modules/pipeline.html
#==============================================================================

from sklearn.pipeline import Pipeline, make_union, Parallel
from h2o.transforms.preprocessing import H2OScaler
from h2o.transforms.decomposition import H2OPCA
from h2o.estimators.random_forest import H2ORandomForestEstimator
#from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
#from h2o.estimators.gbm import H2OGradientBoostingEstimator

#pipeline.fit?
#H2OPrincipalComponentAnalysisEstimator?
#sklearn.pipeline.Parallel?
#sklearn.pipeline.make_union?

iris_df = h2o.import_file(path=iris_path)
iris_df.as_data_frame().head(5)

pipeline = Pipeline([("standardize", H2OScaler()),
                     ("pca", H2OPCA(k=2)),
                     ("drf", H2ORandomForestEstimator(ntrees=200))])
    
pipeline.fit(iris_df[:4],iris_df[4])


#==============================================================================
# scikit-learn style hyperparameter grid search using k-fold cross
#validation
#==============================================================================

#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from h2o.cross_validation import H2OKFold
from h2o.model.regression import h2o_r2_score
from sklearn.metrics.scorer import make_scorer


custom_cv = H2OKFold(iris_df, n_folds=5, seed=42)

params = {"standardize__center": [True, False],
          "standardize__scale": [True, False],
          "pca__k": [2,3],
          "drf__ntrees":[100,200]}

pipeline = Pipeline([("standardize", H2OScaler()),
                     ("pca", H2OPCA()),
                     ("drf", H2ORandomForestEstimator())])

random_search = RandomizedSearchCV(pipeline, params,
                                   n_iter=5,
                                   scoring=make_scorer(h2o_r2_score),
                                   cv=custom_cv,
                                   random_state=42,
                                   n_jobs=1)
iris_df.as_data_frame().head(5)
random_search.fit(X=iris_df[:4], y=iris_df[4])




