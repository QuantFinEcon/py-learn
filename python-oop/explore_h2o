
import h2o
import numpy as np

dir(h2o)

# open FLOW at http://localhost:54321/flow/index.html
h2o.init(ip="127.0.0.1", port=54321)

help(h2o.estimators.glm.H2OGeneralizedLinearEstimator)
h2o.estimators.glm.H2OGeneralizedLinearEstimator?

#h2o.demo("glm")

#l=[[1, 2, 3], ['a','b','c'], [0.1, 0.2, 0.3]]
#df = h2o.H2OFrame(zip(*l))
#
#
#df = h2o.H2OFrame.from_python({"A":1,
#                               "B":[1,2,3]})

df = h2o.H2OFrame.from_python(np.random.randn(100,4).tolist(), column_names=list('ABCD'))
df.as_data_frame().head(5)
df.describe(chunk_summary=True)
df['A']
df['A'].as_data_frame()

df[0:2].as_data_frame()
df[0:5,:].as_data_frame()
df[ df["B"] > 0, :].as_data_frame()

df.isna()

df.apply(lambda row: row.sum(), axis=1)

df['A'].mean(skipna=True)
df['A'].hist(plot=True)















#==============================================================================
#  POJO is a standalone Java class with no dependencies on the full H2O stack 
# (only the h2o-genmodel.jar file, which defines the POJO interface).
#==============================================================================

import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator

h2o.init()

h2o_df = h2o.import_file(path = "C:/xxx.csv",
                         parse=True,header=0,sep=",")

type(h2o_df)

h2o_df.describe()
h2o_df.columns
h2o_df.types
h2o_df.as_data_frame()
h2o_df.as_data_frame()
type(_)



h2o.import_file?
h2o.parse_setup?

cluster_estimator = H2OKMeansEstimator(k=2)

cluster_estimator.train?
cluster_estimator.train(x=['size','numberofrecords'], 
                        y='FilteredFilename',
                        training_frame=h2o_df,
                        verbose=False)

h2o.download_pojo(cluster_estimator)

cluster_estimator


#h2o_df['CAPSULE'] = h2o_df['CAPSULE'].asfactor()
#model = h2o.glm(y = "CAPSULE",
#                x = ["AGE", "RACE", "PSA", "GLEASON"],
#                training_frame = h2o_df,
#                family = "binomial")
#h2o.download_pojo(model)

#==============================================================================
# binary logistic
#==============================================================================

help(h2o.glm)
h2o.demo("glm")


from h2o.estimators.glm import H2OGeneralizedLinearEstimator

h2o_df['FilteredFilename'] = h2o_df['FilteredFilename'].asfactor()
h2o_df.types
h2o_df.columns

train, test = h2o_df.split_frame(ratios=[0.70])


H2OGeneralizedLinearEstimator?
logistic_glm = H2OGeneralizedLinearEstimator(family="binomial")
logistic_glm.train(x=["size","numberofrecords"], 
                   y='FilteredFilename',
                   training_frame=train)
logistic_glm.show()

predictions = logistic_glm.predict(test)
predictions.show()


#==============================================================================
# multi class logistic
#==============================================================================

multinomial_glm = H2OGeneralizedLinearEstimator(family="multinomial")
multinomial_glm.train(x=["size","numberofrecords"], 
                   y='FilteredFilename',
                   training_frame=train)

multinomial_glm.show()

predictions = multinomial_glm.predict(test)
predictions.show()

#==============================================================================
# linear regression
#==============================================================================

glm = H2OGeneralizedLinearEstimator(family="gaussian", link='identity')
glm.train(x=["size","numberofrecords"], 
                   y='FilteredFilename',
                   training_frame=train)

multinomial_glm.show()

predictions = multinomial_glm.predict(test)
predictions.show()


#==============================================================================
# deep learning autoencoder for anomaly detection
#==============================================================================


h2o_df['FilteredFilename'] = h2o_df['FilteredFilename'].asfactor()
h2o_df.types
h2o_df.columns
h2o_df = h2o_df[['FilteredFilename','size','numberofrecords']]

train, test = h2o_df.split_frame(ratios=[0.7])
train.shape
test.shape


dir(h2o.estimators.deeplearning)
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

anomaly_model = H2OAutoEncoderEstimator(activation = "Tanh",
                                        hidden = [50,50,50],
                                        sparse=True,
                                        l1=1e-4,
                                        epochs=100)

anomaly_model.train(x=train.columns, training_frame=train)


anomaly_model.


recon_error = anomaly_model.anomaly(test)
recon_error.shape

recon_error.as_data_frame()
# reconstructured predictions far away from original ==> outliers?
(recon_error.as_data_frame()>0.099)
test.as_data_frame().loc[,:]


# Note: Testing = Reconstructing the test dataset
test_recon = anomaly_model.predict(test)
predict=test_recon.as_data_frame()
list(predict.columns.values)



#==============================================================================
# grid search with H2o
#==============================================================================

import pandas as pd

#iris_data_path = "http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris.csv" # load demonstration data
iris_df = h2o.import_file(path='C:/Users/1580873/Desktop/Completed_Developments/H2O/iris.csv')

iris_df.as_data_frame().head()

ntrees_opt = [5, 10, 15]
max_depth_opt = [2, 3, 4]
learn_rate_opt = [0.1, 0.2]

hyper_parameters = {"ntrees": ntrees_opt, "max_depth":max_depth_opt,
                    "learn_rate":learn_rate_opt}

from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator

gs = H2OGridSearch(H2OGradientBoostingEstimator(distribution="multinomial"), 
                   hyper_params=hyper_parameters)

iris_df.columns
gs.train?
gs.train(x=range(0,iris_df.ncol-1), y=iris_df.ncol-1, 
         training_frame=iris_df, nfolds=10)

gs.get_grid?
gs.get_grid("logloss", decreasing=False)


#==============================================================================
# integrate with scikit-learn http://scikit-learn.org/stable/modules/pipeline.html
#==============================================================================

from sklearn.pipeline import Pipeline, make_union
from h2o.transforms.preprocessing import H2OScaler
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator


h2o.__PROGRESS_BAR__=True

pipeline = Pipeline([("standardize", H2OScaler()),
                     ("pca", H2OPrincipalComponentAnalysisEstimator(k=2,impute_missing=True)),
                     ("gbm", H2OGradientBoostingEstimator(distribution="multinomial"))])

pipeline.fit?
H2OPrincipalComponentAnalysisEstimator?

pipeline.fit(X=iris_df[:4], y=iris_df[4])


import sklearn.pipeline
dir(sklearn.pipeline)

sklearn.pipeline.Parallel?
sklearn.pipeline.make_union?

from sklearn.decomposition import PCA, TruncatedSVD
make_union(PCA(), TruncatedSVD())    # doctest: +NORMALIZE_WHITESPACE

pipeline = Pipeline([(make_union(PCA(), TruncatedSVD())),
                     ("standardize", H2OScaler()),
                     ("pca", H2OPrincipalComponentAnalysisEstimator(k=2,impute_missing=True)),
                     ("gbm", H2OGradientBoostingEstimator(distribution="multinomial"))])



FeatureUnion(n_jobs=1,
       transformer_list=[('pca',
                          PCA(copy=True, iterated_power='auto',
                              n_components=None, random_state=None,
                              svd_solver='auto', tol=0.0, whiten=False)),
                         ('truncatedsvd',
                          TruncatedSVD(algorithm='randomized',
                          n_components=2, n_iter=5,
                          random_state=None, tol=0.0))],
       transformer_weights=None)



#==============================================================================
# scikit-learn style hyperparameter grid search using k-fold cross
#validation
#==============================================================================

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from h2o.cross_validation import H2OKFold
from h2o.model.regression import h2o_r2_score
from sklearn.metrics.scorer import make_scorer








#==============================================================================
# 
#==============================================================================

import sklearn
dir(sklearn.model_selection) 


dir(sklearn.pipeline)
http://michelleful.github.io/code-blog/2015/06/20/pipelines/
http://h2o-release.s3.amazonaws.com/h2o/rel-wolpert/8/docs-website/h2o-py/docs/index.html







