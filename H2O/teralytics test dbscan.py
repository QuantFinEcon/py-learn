import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

#==============================================================================
# for test
#==============================================================================
df=pd.read_csv('reduced.csv',sep=',')

df.shape # 2,228,820 obs
df.head
df.columns
for i in df.columns: print df[i].dtype

# data type fix
df.uid = df.uid.astype('string')
df.floor = df.floor.astype('category')
df.timestamp = pd.to_datetime(df.timestamp)
df.set_index(df.timestamp,inplace=True)
df = df.drop('timestamp',1)
df = df.sort_index()

# drop duplicates i.e. no movement <=> same (time,x,y)
df.loc[df.duplicated(),]
df.query("uid == '5e7b40e1'")
df = df.drop_duplicates()

# 1 sec compression

#b=df.index
#pd.unique(b.strftime('%Y-%m-%d %H:%M:%S')).shape
df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
df.loc[df.duplicated(),].shape
df.index = pd.to_datetime(df.index)

# time delta in terms of seconds, independent from clock measure
delta = df.index.to_series().apply(lambda x: x-df.index[0])
delta.dtype
df = df.assign(delta = delta.apply(lambda x: x.seconds))
df = df.assign(time = df.index)

# USING ASSUMPTIONS A and B
df.shape
#df = df.drop('position',1)
df['position'] = df[['x', 'y','delta']].apply(tuple, axis=1)


# http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
# https://blog.dominodatalab.com/topology-and-density-based-clustering/
from sklearn.cluster import DBSCAN
pd.set_option('display.precision', 4)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 30)

#xx = df.sample(30)
#xx.index = xx[['time','uid']].apply(tuple,axis=1)
df.index = df[['time','uid']].apply(tuple,axis=1)

X1 = df.loc[df.floor==1, ['x','y','delta']]
X2 = df.loc[df.floor==2, ['x','y','delta']]
X3 = df.loc[df.floor==3, ['x','y','delta']]

db = DBSCAN(eps=1, metric='euclidean', min_samples=2, algorithm = 'auto', n_jobs=3)

db.fit(X1)
labels1 = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels1)) - (1 if -1 in labels1 else 0) # -1 labels are outliers
print('Estimated number of clusters FLOOR 1: %d' % n_clusters_)
X1 = X1.assign(DBSCAN_label = labels1)

db.fit(X2)
labels2 = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels2)) - (1 if -1 in labels2 else 0) # -1 labels are outliers
print('Estimated number of clusters FLOOR 2: %d' % n_clusters_)
ad = np.max(labels1)
l2=[x+ad if x >= 0 else x for x in labels2]
X2 = X2.assign(DBSCAN_label = l2)

db.fit(X3)
labels3 = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels3)) - (1 if -1 in labels3 else 0) # -1 labels are outliers
print('Estimated number of clusters FLOOR 3: %d' % n_clusters_)
ad = np.max(l2)
l3=[x+ad if x >= 0 else x for x in labels3]
X3 = X3.assign(DBSCAN_label = l3)

XX=pd.concat([X1,X2],axis=0)
XX=pd.concat([XX,X3],axis=0)

for i in range(10): print XX.loc[XX.DBSCAN_label==i,]

metup = XX.loc[XX.DBSCAN_label!=-1,]

metup.to_csv('meetings.csv')






#==============================================================================
# MISC
#==============================================================================

# descriptive stats
pd.unique(df.floor)
df.loc[df['floor']==1,:].describe()
df.loc[df['floor']==2,:].describe()
df.loc[df['floor']==3,:].describe()

pd.unique(df.uid).shape
print min(df.x); print max(df.x)
print min(df.y); print max(df.y)
df.x.hist()
df.y.hist()

# time delta 
tt=pd.unique(df.index)
tt.shape

g=df.groupby(df.index)
g1=g.uid.count()
ax=g1.plot()
ax.set(ylabel='count of people')
plt.show()

t1=pd.DataFrame(tt)
t1.columns= "delta"
t1=(t1-t1.shift(1))[1:]
t1.iloc[:,0].hist()
t1 = t1.iloc[:,0].apply(lambda x: 100 * x.microseconds)
t1 = t1/1000000 # convert to seconds
ax = t1.loc[t1<1,].hist(bins=np.linspace(0,1,50)); ax.set(ylabel="frequency",xlabel='time between data capture (in seconds)',title = 'less than 1 second distribution')
ax = t1.loc[t1>1,].hist(bins=range(int(min(t1)), int(max(t1))+1, 1)); ax.set(ylabel="frequency",xlabel='time between data capture (in seconds)',title='beyond 1 second distribution')

# transform to cross sectional for INTERPOLATION
df['position'] = df[['x', 'y','floor','delta']].apply(tuple, axis=1)
X = df #backup copy
X = X.pivot_table(values='position',index='time',columns='uid',aggfunc='first')
who = X.count(0)
#example
X.loc[X["ff3f94a4"].notnull(),"ff3f94a4"]
k=X["ff3f94a4"].notnull()
k.index = range(k.shape[0])
ii=k.loc[k,].index
bb=(ii.to_series()-ii.to_series().shift(1))[1:,]
max(bb)
bb.loc[bb==8,]
X.loc[X.index[46415]:,"ff3f94a4"] # 8 SECOND MISSING POSITIONS!! NEED INTERPOLATE! 




     
     