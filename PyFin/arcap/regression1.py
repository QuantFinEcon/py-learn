filename = 'risk tickers.xlsx'
xls = pd.ExcelFile(filename)
sht = xls.sheet_names
for i in sht: print i

df = pd.read_excel(filename,sheetname='Iron Steel')
#print the column names
print df.columns

#get the values for a given column
FLDS = df['Tickers'].values
for i in FLDS: print i
FLDS.shape
FLDS = FLDS.tolist()
FLDS = [x for x in FLDS if str(x) != 'nan']

len(FLDS)

setup_bbg()
start_date = '1/1/2010'
starting = time.time()
sids = mgr[FLDS]
x = sids.get_historical("PX_LAST", start_date, date.today()) #ISO FX
x.backup = x.copy
print(time.time()-starting)


# fill na
x = x.fillna(method="ffill") 
#x = x.fillna(method="bfill") # hindsight biases
x.tail()

x.plot()
x.apply(normalise_max_min).plot()


#==============================================================================
# R^2 explanation
#==============================================================================
from sklearn import datasets, linear_model
regr = linear_model.LinearRegression(normalize = True, n_jobs = -1)

x.T.tail()

X = df_dropna(x.drop('ISIXSTSC Index',axis=1),axis=0)
Y = df_dropna(x['ISIXSTSC Index'].to_frame(),axis=0)
merged = pd.concat([Y,X],axis=1,join='inner') #merged = merged.dropna(axis=0)
X = merged.iloc[:,1:]
Y = merged.iloc[:,0]

# regression model outputs
regr.fit(X, Y) # scipy.linalg stored result in global envir
regr.intercept_
regr.coef_

#plot residuals
regr_res = Y - regr.predict(X)
regr_res.plot()

#merge for plotting
if len(regr.predict(X)) == Y.shape[0]: 
    Y1 = pd.concat([Y,pd.DataFrame(regr.predict(X),index=Y.index)],axis=1)
Y1.columns = ['Actual','Predicted']
Y1.head()
Y1.plot()

# plot regression coefficients
plot_coeff(regr.coef_)

for i in r1: print i.get_x()


#==============================================================================
# STATS TOOLS
#==============================================================================

# test stationary pairs - cointegration
import statsmodels.tsa.stattools as tsa
import statsmodels.graphics.tsaplots as tsa_plots

tsa.adfuller(regr_res)

setup_bbg()
start_date = '1/1/2010'
starting = time.time()
sids = mgr['GOOG US Equity']
x = sids.get_historical("PX_LAST", start_date, date.today()) #ISO FX
Y = x.fillna(method="ffill") 


acf_ = tsa_plots.plot_acf(x=Y,alpha =.05, use_vlines=True, lags=100, unbiased=True)
acf_ = tsa_plots.plot_acf(x=log_return(Y),alpha =.05, use_vlines=True, lags=100, unbiased=True)

ret=log_return(Y)
ret.describe()
ret.plot()
#ACF
acf_, ci, Q, pvalue = tsa.acf(ret, nlags=30, alpha=.05, qstat=True) # use FFT is long ts
tsa_plots.plot_acf(x=ret,alpha =.05, use_vlines=True, lags=30, unbiased=True)
out=np.column_stack((acf_,ci))
in_cf=map(lambda x : x[0]>=x[1] and x[0]<=x[2], out)

#PACF
pacf_, ci = tsa.pacf(ret, nlags=30, alpha=.05) # use FFT is long ts
tsa_plots.plot_pacf(x=ret,alpha =.05, use_vlines=True, lags=30)
out=np.column_stack((pacf_,ci))
in_cf=map(lambda x : x[0]>=x[1] and x[0]<=x[2], out)

X




#==============================================================================
# decomposition of features - PCA with Varimax roration
#==============================================================================

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    from scipy import eye, asarray, dot, sum, svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)

from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)



#==============================================================================
# example regression
#==============================================================================
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#==============================================================================
# stats model regression / ROLLING
#==============================================================================
x.shape
y=x.iloc[:,0]
X=x.iloc[:,1:]
model = pd.stats.ols.MovingOLS(y=y, x=X, 
                               window_type='rolling', window=100, intercept=True)



#r.agg         r.apply       r.count       r.exclusions  r.max         r.median      r.name        r.skew        r.sum
#r.aggregate   r.corr        r.cov         r.kurt        r.mean        r.min         r.quantile    r.std         r.var

x.rolling(window=62,center=False).mean()

x.rolling(window=62,center=False).apply(normalise_max_min,axis=0)

R = x.rolling(window=62,min_periods=20,center=False)
R.mean()

R.agg({'result1' : np.sum,'result2' : np.mean})

R.apply(lambda x: np.std(x))


C=R.corr(x[x.columns.values[0]],pairwise=True)
C.columns
R.apply(corr,x[x.columns.values[0]],pairwise=True)
R.corr(pairwise=True)

import inspect
print inspect.getsource(clustered_corr)
print inspect.getsource(R.corr)
R.corr??
R.apply??

C.columns
C.tail(2)
C.iloc[:,1].plot()
x.shape


# 2 arguments mapper with lambda
f = lambda (x, y): (x+y, x-y)
t1=x.iloc[:,0]
t2=x.iloc[:,1]
from itertools import repeat
zz=map(f, zip(t1,t2))

h=pd.DataFrame()
h['a'],h['b']=zip(*zz)
h.tail()
type(h)

#lambda (x,y) for correlation
from scipy.stats.stats import pearsonr
pearsonr(t1,t2,)
f = lambda (x, y): (x+y, x-y)
t1=x.iloc[:,0]
t2=x.iloc[:,1]
from itertools import repeat
zz=map(f, zip(t1,t2))
zz=x.iloc[:,:2].apply(tuple,axis=1)

h=pd.DataFrame()
h['a'],h['b']=zip(*zz)
h.tail()
type(h)

plot_pdf_level_one(C,nrow=3,ncol=3,total=C.shape[1],sz=(10,8),filename="rolling_corr.pdf")



#==============================================================================
# ROLLING OLS REGRESSION PAIRWISE R^2 take TOP 3
#==============================================================================

y = x['ISIXSTSC Index'].to_frame()
x = x.drop(['ISIXSTSC Index'],axis=1)
y.shape
x.shape
pairwise_R2(y,x,start_date='2015',lookback=21,min_pd=5,sz=(15,15),filename='pairwise_rollingR2.pdf')


#==============================================================================
# MANUAL PCA
#==============================================================================
X = df_dropna(x)
eigenvals, components = np.linalg.eig(np.cov(X.transpose()))

vr_components = pretty_matrix(mat=varimax(components[:,:3]),digits=7)
pd.DataFrame(vr_components,index=mgr[X.columns].NAME)


# find index of coeff != 0 in components
ID_tup = zip(*np.where(vr_components != 0))
ID = pd.DataFrame(ID_tup).iloc[:,1].values
mgr[X.columns[ID]].NAME



#==============================================================================
# AUTO PCA
#==============================================================================
pca = PCA(n_components=3) # tol for singular SVD
X = df_dropna(x)
fit = pca.fit(X)

trans_X = pca.transform(X)
trans_X = pd.DataFrame(trans_X, index = X.index, 
                       columns=["PC"+str(i+1) for i in xrange(trans_X.shape[1])])
trans_X.plot() # plot first 3 components
trans_X.corr()


trans_X.shape
trans_X.describe()
trans_X.head()
y.head()

# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
var1=np.cumsum(np.round(fit.explained_variance_ratio_, decimals=4)*100)
var1
plt.plot(var1)

#np.set_printoptions(threshold=np.inf)
print(fit.components_.T)
fit.n_features_
fit.n_components_


pairwise_R2(y,trans_X,start_date='2015',
            lookback=21,min_pd=5,sz=(15,15),filename='pairwise_rollingR2 PCA.pdf')


#==============================================================================
# regress 1
#==============================================================================
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_boston()

type(diabetes)
diabetes.keys()
diabetes.data.shape

type(diabetes_X)
type(diabetes.target)

# Use only one feature
diabetes_X = diabetes.data[:, ]
len(diabetes_X)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

regr.score(diabetes_X_train, diabetes_y_train)

 regr.intercept_

diabetes.data

regr.columns

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

