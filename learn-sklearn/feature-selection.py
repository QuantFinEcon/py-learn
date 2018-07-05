
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import sklearn.datasets
#dir(sklearn.datasets)
from sklearn.datasets import (load_boston,
                              load_iris,
                              load_digits,
                              load_diabetes,
                              load_wine,
                              load_breast_cancer,
                              fetch_olivetti_faces,
                              make_friedman1,
                              make_classification,
                              make_blobs,
                              make_circles)


from sklearn.preprocessing import (Binarizer,
                                   FunctionTransformer,
                                   Imputer,
                                   KernelCenterer,
                                   LabelBinarizer,
                                   LabelEncoder,
                                   MaxAbsScaler,
                                   MinMaxScaler,
                                   MultiLabelBinarizer,
                                   Normalizer,
                                   OneHotEncoder,
                                   PolynomialFeatures,
                                   QuantileTransformer,
                                   RobustScaler,
                                   StandardScaler,)

from sklearn.feature_selection import (SelectKBest,
                                       SelectPercentile,
                                       SelectFromModel,
                                       chi2,
                                       mutual_info_classif,
                                       f_classif,
                                       f_regression,
                                       mutual_info_regression,
                                       VarianceThreshold,
                                       RFE, 
                                       RFECV,
                                       
                                       GenericUnivariateSelect,
                                        SelectFdr,
                                        SelectFpr,
                                        SelectFwe,)

from sklearn.feature_extraction import (FeatureHasher,
                                        grid_to_graph,
                                        DictVectorizer,
                                        hashing,
                                        image,
                                        img_to_graph,
                                        text,
                                        stop_words)

from sklearn.pipeline import (Pipeline,
                              make_pipeline,
                              FeatureUnion,
                              make_union)
#                              clone,
#                              delayed,
#                              sparse,
#                              six,
#                              Bunch, #dict like
#                              Memory,
#                              Parallel)

from sklearn.model_selection import (cross_validate,
                                     cross_val_score,
                                     cross_val_predict,
                                     check_cv,
                                     fit_grid_point,
                                     permutation_test_score,
                                     train_test_split,
                                     TimeSeriesSplit,
                                     validation_curve,
                                     KFold,
                                     StratifiedKFold,
                                     RepeatedKFold,
                                     GroupKFold,
                                     GroupShuffleSplit,
                                     ShuffleSplit,
                                     StratifiedShuffleSplit,
                                     LeaveOneOut,
                                     LeaveOneGroupOut,
                                     LeavePOut,
                                     LeavePGroupsOut,
                                     GridSearchCV,
                                     ParameterGrid,
                                     ParameterSampler,
                                     RandomizedSearchCV)

from sklearn.metrics import (explained_variance_score, # regression metrics
                             mean_absolute_error,
                             mean_squared_error,
                             mean_squared_log_error,
                             median_absolute_error,
                             r2_score,
                             
                             adjusted_mutual_info_score, # clusteirng metrics
                             adjusted_rand_score,
                             completeness_score,
                             fowlkes_mallows_score,
                             homogeneity_score,
                             mutual_info_score,
                             normalized_mutual_info_score,
                             v_measure_score,
                             
                             confusion_matrix, # classification metrics
                             classification,
                             classification_report,
                             accuracy_score,
                             average_precision_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             fbeta_score,
                             precision_recall_curve,
                             precision_recall_fscore_support,
                             auc,
                             roc_auc_score,
                             roc_curve,)
                             
"""
base,
brier_score_loss,
calinski_harabaz_score,
cluster,
cohen_kappa_score,
consensus_score,
coverage_error,
euclidean_distances,
get_scorer,
hamming_loss,
hinge_loss,
homogeneity_completeness_v_measure,
jaccard_similarity_score,
label_ranking_average_precision_score,
label_ranking_loss,
log_loss,
make_scorer,
matthews_corrcoef,
pairwise,
pairwise_distances,
pairwise_distances_argmin,
pairwise_distances_argmin_min,
pairwise_fast,
pairwise_kernels,
ranking,
regression,
scorer,
silhouette_samples,
silhouette_score,
zero_one_loss
"""

from sklearn.externals.joblib import (Memory,
                                      Parallel,
                                      Logger,
                                      MemorizedResult,
                                      PrintTime,
                                      backports,
                                      cpu_count,
                                      delayed,
                                      disk,
                                      dump,
                                      effective_n_jobs,
                                      format_stack,
                                      func_inspect,
                                      hash,
                                      hashing,
                                      load,
                                      logger,
                                      memory,
                                      my_exceptions,
                                      numpy_pickle,
                                      numpy_pickle_compat,
                                      numpy_pickle_utils,
                                      parallel,
                                      parallel_backend,
                                      pool,
                                      register_parallel_backend)

from sklearn.ensemble import (ExtraTreesClassifier,
                              ExtraTreesRegressor)
"""
AdaBoostClassifier,
AdaBoostRegressor,
BaggingClassifier,
BaggingRegressor,
BaseEnsemble,
ExtraTreesClassifier,
ExtraTreesRegressor,
GradientBoostingClassifier,
GradientBoostingRegressor,
IsolationForest,
RandomForestClassifier,
RandomForestRegressor,
RandomTreesEmbedding,
VotingClassifier,
_gradient_boosting,
bagging,
base,
forest,
gradient_boosting,
iforest,
partial_dependence,
voting_classifier,
weight_boosting
"""

from sklearn.svm import (SVC, 
                         SVR,
                         LinearSVC, # SVC kernel='linear' but with penalty, implemented in liblinear than libsvm
                         LinearSVR) 

from sklearn.cluster import (KMeans,
                             DBSCAN,
                             FeatureAgglomeration,
                             AgglomerativeClustering?,
                             AffinityPropagation,
Birch,
MeanShift,
MiniBatchKMeans,
SpectralBiclustering,
SpectralClustering,
SpectralCoclustering,
affinity_propagation,
affinity_propagation_,
bicluster,
estimate_bandwidth,
get_bin_seeds,
linkage_tree?,
ward_tree?)

dir(sklearn.cluster.hierarchical)



from sklearn.linear_model import (Lasso, #L1 penalty |lambda|, least abs shrinkage and selection operator
                                  LassoCV,
                                  RandomizedLasso,
                                  lasso_path, #plot coeff_ against shrinkage factor
                                  lars_path, #least angle regression
                                  Ridge, #L2 penalty lambda^2
                                  RidgeCV,
                                  ElasticNet,
                                  ElasticNetCV,
                                  enet_path,
                                  BayesianRidge)
"""
ARDRegression,
BayesianRidge,
ElasticNet,
ElasticNetCV,
Hinge,
Huber,
HuberRegressor,
Lars,
LarsCV,
Lasso,
LassoCV,
LassoLars,
LassoLarsCV,
LassoLarsIC,
LinearRegression,
Log,
LogisticRegression,
LogisticRegressionCV,
ModifiedHuber,
MultiTaskElasticNet,
MultiTaskElasticNetCV,
MultiTaskLasso,
MultiTaskLassoCV,
OrthogonalMatchingPursuit,
OrthogonalMatchingPursuitCV,
PassiveAggressiveClassifier,
PassiveAggressiveRegressor,
Perceptron,
RANSACRegressor,
RandomizedLasso,
RandomizedLogisticRegression,
Ridge,
RidgeCV,
RidgeClassifier,
RidgeClassifierCV,
SGDClassifier,
SGDRegressor,
SquaredLoss,
TheilSenRegressor,
base,
bayes,
cd_fast,
coordinate_descent,
enet_path,
huber,
lasso_stability_path,
least_angle,
logistic,
logistic_regression_path,
omp,
orthogonal_mp,
orthogonal_mp_gram,
passive_aggressive,
perceptron,
randomized_l1,
ransac,
ridge,
ridge_regression,
sag,
sag_fast,
sgd_fast,
stochastic_gradient,
theil_sen
"""

sklearn.linear_model.lasso_path?
sklearn.linear_model.lars_path?



"""
load datasets
"""

boston = load_boston(); print(boston.keys())
iris = load_iris(); print(iris.keys())
digits = load_digits(); print(digits.keys())
diabetes = load_diabetes(); print(diabetes.keys())
wine = load_wine(); print(wine.keys())
breast_cancer = load_breast_cancer(); print(breast_cancer.keys())
faces = fetch_olivetti_faces() #downloading to os.getcwd()

wine.target
breast_cancer.target
diabetes.target
boston.target


def np_to_pandas(ndarray, df_name):
    """
    transform sklearn.datasets to pandas.Dataframe
    @ndarray: np.ndarray
    @df_name: "boston"
    -> output: df_boston
    """
    exec("df_"+ df_name +"= pd.DataFrame("+ df_name +"['data'], columns="+ 
                                        df_name +"['feature_names'])",globals())
    print(eval("df_"+ df_name +".dtypes"))
    print(eval("df_"+ df_name +".describe()"))
    
np_to_pandas(boston,"boston"); df_boston.head()
np_to_pandas(iris,"iris"); df_iris.head()
np_to_pandas(diabetes,"diabetes"); df_diabetes.head()
np_to_pandas(wine,"wine"); df_wine.head()



"""
EDA
"""

def plot_cdf(x):
    """
    Plots Cumulative Distribution Function of an array or Pandas.Series
    @x: array or Pandas.Series
    """
    x=x.sort_values()
    x[len(x)]=x.iloc[-1]
    cum_dist = np.linspace(0.,1.,len(x))
    x_cdf = pd.Series(cum_dist, index=x)
    x_cdf.plot(drawstyle='steps')

#x=plt.hist(df_boston.CHAS,cumulative=True,histtype='bar')

"""
Removing features with low variance
"""

filter_low_variance = VarianceThreshold(threshold=.8*(1-.8))
filter_low_variance.fit_transform(df_boston).shape
df_boston.shape
df_boston.columns.values[~filter_low_variance.get_support()]

plot_cdf(df_boston.CHAS)
plt.plot(df_boston.CHAS)
plot_cdf(df_boston.NOX)
plt.plot(df_boston.NOX)

"""
univariate selection
- apply metric on each column to pick k columns with best score
"""

"""
chi2 - for categorical or sparse data auto treated as discrete
        table of odds-ratio = A->B / NotA->B
"""
X, y = iris.data, iris.target
X.shape
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)
X_new.shape

selector.get_support()
chi2(X, iris.target) #return Chi2 val, pval
df_iris.columns.values[~selector.get_support()]

# top 2 chi2 dependence with target
plt.scatter(iris.target,df_iris.iloc[:,2])
plt.scatter(iris.target,df_iris.iloc[:,3])

# not top 2
plt.scatter(iris.target,df_iris.iloc[:,0])
plt.scatter(iris.target,df_iris.iloc[:,1])


"""
mutual_info_classifi: discrete target
- 0 means independent, higher more dependency
"""
X, y = iris.data, iris.target
X.shape
selector = SelectKBest(mutual_info_classif, k=2)
X_new = selector.fit_transform(X, y)
X_new.shape

selector.get_support()
mutual_info_classif(X, iris.target)

"""
f_classif: Compute the ANOVA F-value between predictor 
            and discrete target variable
F = Explained Sum Variance / Unexplained Sum Variance
"""
X, y = iris.data, iris.target
# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
# 25% most significant features
selector = SelectPercentile(f_classif, percentile=10)
X_new = selector.fit_transform(X, y)
X_new.shape
X.shape[1]/100*10

selector.get_support()
f_classif(X, iris.target) # returns F, pval

"""
scenario of dimensionality curse. features > sample
plot CV perf score against #features used. 
Sometimes, doesn't mean more features always improve perf
"""
# Throw away data, to be in the curse of dimension settings
y = digits.target[:200]
X = digits.data[:200]
X.shape
n_samples = len(y)
# add 200 non-informative features
X = np.hstack((X, 2 * np.random.random((n_samples, 200))))

selector = SelectPercentile(f_classif)
clf = Pipeline([('anova', selector), ('svc', SVC(C=1.0))])

# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    # Compute cross-validation score using 1 CPU
    this_scores = cross_val_score(clf, X, y, n_jobs=1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

cross_val_score?
plt.errorbar(percentiles, score_means, score_stds)
plt.title('Performance of the SVM-Anova varying the percentile of features selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')
plt.axis('tight')
plt.show()

"""
f_regression: univariate selection (each predictor--> numerical target)
    Linear model for testing the "individual" effect of each of many regressors.
    This is done in 2 steps:
        1. The correlation between each regressor and the target is computed, 
            that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) * std(y)).
        2. It is converted to an F score then to a p-value.
"""

X, y = boston.data, boston.target
X.shape
# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(X), 20))
# Add the noisy data to the informative features
X = np.hstack((X, E))
# 25% most significant features
selector = SelectKBest(f_regression, k=3)
X_new = selector.fit_transform(X, y)
X_new.shape

selector.get_support()
f_regression(X, y) # returns F, pval
df_boston.columns.values[12]
plt.scatter(y,df_boston.LSTAT)
df_boston.columns.values[6]
plt.scatter(y,df_boston.AGE)

"""
RFE
Feature ranking with recursive feature elimination.

Given an external estimator that assigns weights to features 
(e.g., the coefficients of a linear model), the goal of 
recursive feature elimination (RFE) is to select features by recursively 
considering smaller and smaller sets of features. 

First, the estimator is trained on the initial set of features and 
the importance of each feature is obtained either through a 
coef_ attribute or through a feature_importances_ attribute. 

Then, the least important features are pruned from current set of features. 

That procedure is recursively repeated on the pruned set until the desired 
number of features to select is eventually reached.
"""

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(X, y)
selector.get_support()
selector.ranking_

plt.scatter(y,X[:,3]) #most linear relationship selected
plt.scatter(y,X[:,0])

estimator.fit(X[:,[0,3]],y)
estimator.predict(X[:,[0,3]])
estimator.coef_ # 3 has larger coef_ hence rank 1st


X, y = boston.data, boston.target
estimator = SVR(kernel="linear") # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(X, y)
selector.get_support()
selector.ranking_

#1
plt.scatter(y,X[:,5]) # most linear related to y
#2
plt.scatter(y,X[:,4])

X, y = boston.data, boston.target
estimator = SVR(kernel='poly', degree=2) # more computationally expensive
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(X, y)
selector.get_support()
selector.ranking_

# reshaping for images
X, y = digits.images.reshape((len(digits.images), -1)), digits.target
digits.images.shape # 8 2D arrays
X.shape

#Display the first digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# first 4 digits
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking --> CENTRE AREA HIGHER RANK
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()


"""
RFECV
    automatic tuning of the number of features selected with cross-validation
"""

make_classification?
"""
Generate a random n-class classification problem.

This initially creates clusters of points normally distributed (std=1)
about vertices of an `n_informative`-dimensional hypercube with sides of
length `2*class_sep` and assigns an equal number of clusters to each
class. It introduces interdependence between these features and adds
various types of further noise to the data.

Prior to shuffling, `X` stacks a number of these primary "informative"
features, "redundant" linear combinations of these, "repeated" duplicates
of sampled features, and arbitrary noise for and remaining features.
"""
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=2, random_state=0)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications. True Positive. 
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

"""
LASSO - L1 penalty |lambda|, least abs shrinkage and selection operator
    lasso_path - which predictor's coeff survive and remain high the longest
                    as penalty increases? 

For a good choice of alpha, the Lasso can fully recover the exact set of 
non-zero variables using only few observations, provided certain 
specific conditions are met. 

In particular, the number of samples should be “sufficiently large”, 
or L1 models will perform at random, where “sufficiently large” depends on 
the number of non-zero coefficients, the logarithm of the number of features, 
the amount of noise, the smallest absolute value of non-zero coefficients, 
and the structure of the design matrix X. 

In addition, the design matrix must display certain specific properties, 
such as not being too correlated. 
"""

x = diabetes.data
y = diabetes.target
x.shape
X = x/x.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

# Compute paths

eps = 5e-3  # the smaller it is the longer is the path

lasso_path?
"""
Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data to avoid
    unnecessary memory duplication. If ``y`` is mono-output then ``X``
    can be sparse.

y : ndarray, shape (n_samples,), or (n_samples, n_outputs)
    Target values

eps : float, optional
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``

n_alphas : int, optional
    Number of alphas along the regularization path

alphas : ndarray, optional
    List of alphas where to compute the models.
    If ``None`` alphas are set automatically

precompute : True | False | 'auto' | array-like
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

Xy : array-like, optional
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

coef_init : array, shape (n_features, ) | None
    The initial values of the coefficients.

verbose : bool or integer
    Amount of verbosity.

return_n_iter : bool
    whether to return the number of iterations or not.

positive : bool, default False
    If set to True, forces coefficients to be positive.
    (Only allowed when ``y.ndim == 1``).

**params : kwargs
    keyword arguments passed to the coordinate descent solver.

Returns
-------
alphas : array, shape (n_alphas,)
    The alphas along the path where models are computed.

coefs : array, shape (n_features, n_alphas) or             (n_outputs, n_features, n_alphas)
    Coefficients along the path.

dual_gaps : array, shape (n_alphas,)
    The dual gaps at the end of the optimization for each alpha.

n_iters : array-like, shape (n_alphas,)
    The number of iterations taken by the coordinate descent optimizer to
    reach the specified tolerance for each alpha.
"""
print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X, y, eps, positive=True, fit_intercept=False)

#print("Computing regularization path using the elastic net...")
#alphas_enet, coefs_enet, _ = enet_path(
#    X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)
#
#print("Computing regularization path using the positive elastic net...")
#alphas_positive_enet, coefs_positive_enet, _ = enet_path(
#    X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

# Display results
plt.figure(1)
ax = plt.gca()

# https://matplotlib.org/tutorials/colors/colormaps.html
# dir(plt.cm)
colormap = plt.cm.Dark2
colors = [colormap(i) for i in range(len(coefs_lasso))]

# -np.log10 --> plot will be normalised as coeff_ change more 
#   when penalty is higher with smaller alpha
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend( (l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
plt.axis('tight')


plt.figure(2)
ax = plt.gca()

# https://matplotlib.org/tutorials/colors/colormaps.html
# dir(plt.cm)
colormap = plt.cm.Dark2
colors = [colormap(i) for i in range(len(coefs_lasso))]

# -np.log10 --> plot will be normalised as coeff_ change more 
#   when penalty is higher with smaller alpha
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)
lasso_cv = LassoCV(eps).fit(X,y)
lasso_cv.coef_
lasso_cv.alpha_
neg_log_alphas_lasso_cv = -np.log10(lasso_cv.alpha_)

for coef_l, coef_pl, c, l in zip(coefs_lasso, 
                                 coefs_positive_lasso, 
                                 colors, 
                                 diabetes.feature_names):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c, label=l)
    l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

plt.axvline(neg_log_alphas_lasso_cv,c='r',linestyle='--',lw=0.5)
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend(loc='lower left')
plt.axis('tight')

"""
Similar to REF, whereby an estimator model coef_ or feature_importance_ are used to rank,
instead of specifying #features and keeping the top rank,
specify a threshold and features significance below threshold
are deem unimportant and dropped

SelectFromModel?
Init signature: SelectFromModel(estimator, threshold=None, prefit=False, norm_order=1)
Docstring:     
Meta-transformer for selecting features based on importance weights.

Parameters
----------
estimator : object
    The base estimator from which the transformer is built.
    This can be both a fitted (if ``prefit`` is set to True)
    or a non-fitted estimator. The estimator must have either a
    ``feature_importances_`` or ``coef_`` attribute after fitting.

threshold : string, float, optional default None
    The threshold value to use for feature selection. Features whose
    importance is greater or equal are kept while the others are
    discarded. If "median" (resp. "mean"), then the ``threshold`` value is
    the median (resp. the mean) of the feature importances. A scaling
    factor (e.g., "1.25*mean") may also be used. If None and if the
    estimator has a parameter penalty set to l1, either explicitly
    or implicitly (e.g, Lasso), the threshold used is 1e-5.
    Otherwise, "mean" is used by default.

prefit : bool, default False
    Whether a prefit model is expected to be passed into the constructor
    directly or not. If True, ``transform`` must be called directly
    and SelectFromModel cannot be used with ``cross_val_score``,
    ``GridSearchCV`` and similar utilities that clone the estimator.
    Otherwise train the model using ``fit`` and then ``transform`` to do
    feature selection.

norm_order : non-zero int, inf, -inf, default 1
    Order of the norm used to filter the vectors of coefficients below
    ``threshold`` in the case where the ``coef_`` attribute of the
    estimator is of dimension 2.

Attributes
----------
estimator_ : an estimator
    The base estimator from which the transformer is built.
    This is stored only when a non-fitted estimator is passed to the
    ``SelectFromModel``, i.e when prefit is False.

threshold_ : float
    The threshold value used for feature selection.


LinearSVC?
Init signature: LinearSVC(penalty='l2', loss='squared_hinge', 
                          dual=True, tol=0.0001, C=1.0, multi_class='ovr', #one-over-the-rest
                          fit_intercept=True, intercept_scaling=1, 
                          class_weight=None, verbose=0, random_state=None, max_iter=1000)

Parameters
----------
penalty : string, 'l1' or 'l2' (default='l2')
    Specifies the norm used in the penalization. The 'l2'
    penalty is the standard used in SVC. The 'l1' leads to ``coef_``
    vectors that are sparse.

loss : string, 'hinge' or 'squared_hinge' (default='squared_hinge')
    Specifies the loss function. 'hinge' is the standard SVM loss
    (used e.g. by the SVC class) while 'squared_hinge' is the
    square of the hinge loss.

dual : bool, (default=True)
    Select the algorithm to either solve the dual or primal
    optimization problem. Prefer dual=False when n_samples > n_features.

tol : float, optional (default=1e-4)
    Tolerance for stopping criteria.

C : float, optional (default=1.0)
    Penalty parameter C of the error term.

multi_class : string, 'ovr' or 'crammer_singer' (default='ovr')
    Determines the multi-class strategy if `y` contains more than
    two classes.
    ``"ovr"`` trains n_classes one-vs-rest classifiers, while
    ``"crammer_singer"`` optimizes a joint objective over all classes.
    While `crammer_singer` is interesting from a theoretical perspective
    as it is consistent, it is seldom used in practice as it rarely leads
    to better accuracy and is more expensive to compute.
    If ``"crammer_singer"`` is chosen, the options loss, penalty and dual
    will be ignored.

fit_intercept : boolean, optional (default=True)
    Whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be already centered).

intercept_scaling : float, optional (default=1)
    When self.fit_intercept is True, instance vector x becomes
    ``[x, self.intercept_scaling]``,
    i.e. a "synthetic" feature with constant value equals to
    intercept_scaling is appended to the instance vector.
    The intercept becomes intercept_scaling * synthetic feature weight
    Note! the synthetic feature weight is subject to l1/l2 regularization
    as all other features.
    To lessen the effect of regularization on synthetic feature weight
    (and therefore on the intercept) intercept_scaling has to be increased.

class_weight : {dict, 'balanced'}, optional
    Set the parameter C of class i to ``class_weight[i]*C`` for
    SVC. If not given, all classes are supposed to have
    weight one.
    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

verbose : int, (default=0)
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in liblinear that, if enabled, may not work
    properly in a multithreaded context.

random_state : int, RandomState instance or None, optional (default=None)
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

max_iter : int, (default=1000)
    The maximum number of iterations to be run.

Attributes
----------
coef_ : array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    ``coef_`` is a readonly property derived from ``raw_coef_`` that
    follows the internal memory layout of liblinear.

intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
    Constants in decision function.
"""

X, y = iris.data, iris.target
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
model.get_support()
X_new = model.transform(X)
X_new.shape

"""
Tree-based feature selection - feature_importances_

ExtraTreesClassifier?
ExtraTreesClassifier(n_estimators=10, criterion='gini', 
                     max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                     min_weight_fraction_leaf=0.0, max_features='auto', 
                     max_leaf_nodes=None, min_impurity_decrease=0.0, 
                     min_impurity_split=None, bootstrap=False, oob_score=False, 
                     n_jobs=1, random_state=None, verbose=0, 
                     warm_start=False, class_weight=None)

Parameters
----------
n_estimators : integer, optional (default=10)
    The number of trees in the forest.

criterion : string, optional (default="gini")
    The function to measure the quality of a split. Supported criteria are
    "gini" for the Gini impurity 
    and "entropy" for the information gain.

max_features : int, float, string or None, optional (default="auto")
    The number of features to consider when looking for the best split:

    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a percentage and
      `int(max_features * n_features)` features are considered at each
      split.
    - If "auto", then `max_features=sqrt(n_features)`. <=======
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.

    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.

max_depth : integer or None, optional (default=None)
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

min_samples_split : int, float, optional (default=2)
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a percentage and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

    .. versionchanged:: 0.18
       Added float values for percentages.

min_samples_leaf : int, float, optional (default=1)
    The minimum number of samples required to be at a leaf node:

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a percentage and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for percentages.

min_weight_fraction_leaf : float, optional (default=0.)
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.

max_leaf_nodes : int or None, optional (default=None)
    Grow trees with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.

min_impurity_decrease : float, optional (default=0.)
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following:

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.

    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.

    .. versionadded:: 0.19

bootstrap : boolean, optional (default=False)
    Whether bootstrap samples are used when building trees.

oob_score : bool, optional (default=False)
    Whether to use out-of-bag samples to estimate
    the generalization accuracy.

n_jobs : integer, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    If -1, then the number of jobs is set to the number of cores.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

verbose : int, optional (default=0)
    Controls the verbosity of the tree building process.

warm_start : bool, optional (default=False)
    When set to ``True``, reuse the solution of the previous call to fit
    and add more estimators to the ensemble, otherwise, just fit a whole
    new forest.

class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None)
    Weights associated with classes in the form ``{class_label: weight}``.
    If not given, all classes are supposed to have weight one. For
    multi-output problems, a list of dicts can be provided in the same
    order as the columns of y.

    Note that for multioutput (including multilabel) weights should be
    defined for each class of every column in its own dict. For example,
    for four-class multilabel classification weights should be
    [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
    [{1:1}, {2:5}, {3:1}, {4:1}].

    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``

    The "balanced_subsample" mode is the same as "balanced" except that weights are
    computed based on the bootstrap sample for every tree grown.

    For multi-output, the weights of each column of y will be multiplied.

    Note that these weights will be multiplied with sample_weight (passed
    through the fit method) if sample_weight is specified.

Attributes
----------
estimators_ : list of DecisionTreeClassifier
    The collection of fitted sub-estimators.

classes_ : array of shape = [n_classes] or a list of such arrays
    The classes labels (single output problem), or a list of arrays of
    class labels (multi-output problem).

n_classes_ : int or list
    The number of classes (single output problem), or a list containing the
    number of classes for each output (multi-output problem).

feature_importances_ : array of shape = [n_features]
    The feature importances (the higher, the more important the feature).

n_features_ : int
    The number of features when ``fit`` is performed.

n_outputs_ : int
    The number of outputs when ``fit`` is performed.

oob_score_ : float
    Score of the training dataset obtained using an out-of-bag estimate.

oob_decision_function_ : array of shape = [n_samples, n_classes]
    Decision function computed with out-of-bag estimate on the training
    set. If n_estimators is small it might be possible that a data point
    was never left out during the bootstrap. In this case,
    `oob_decision_function_` might contain NaN.
    
values for the parameters controlling the size of the trees
(e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
unpruned trees which can potentially be very large on some data sets. To
reduce memory consumption, the complexity and size of the trees should be
controlled by setting those parameter values.    
    
implements a meta estimator that fits a number of
randomized decision trees (a.k.a. extra-trees) on various sub-samples
of the dataset and use averaging to improve the predictive accuracy
and control over-fitting.
"""
X, y = iris.data, iris.target
X.shape

clf = ExtraTreesClassifier(n_estimators=30,
                           criterion = 'gini',
                           min_samples_split=2,
                           min_samples_leaf=1,
                           min_impurity_decrease=0,
                           oob_score=False,                           
                           verbose=1)
clf = clf.fit(X, y)
clf.feature_importances_ # last 2 more important
clf.estimators_
clf.n_classes_
set(y)

model = SelectFromModel(clf, prefit=True)
model.get_support()
X_new = model.transform(X)
X_new.shape

"""
Pixel importances with a parallel forest of trees
"""
#faces.images[399].shape (64,64) per face. there is 400 faces. 
#faces.images.shape (400,64,64) --> (400,-1)
X = faces.images.reshape((len(faces.images), -1)) #flatten 64x64 to 4096
X.shape
y = faces.target
faces.images[0][0] == X[0][:64]
faces.images[0][1] == X[0][64:128]

# Build a forest and compute the pixel importances
forest = ExtraTreesClassifier(n_estimators=1000,
                              max_features=128,
                              criterion='gini',
                              min_samples_leaf=1,
                              min_samples_split=2,
                              n_jobs=-1,
                              random_state=0,
                              verbose=1)

forest.fit(X, y)
importances = forest.feature_importances_
importances = importances.reshape(faces.images[0].shape) # shape (64,64)
importances.shape

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
plt.show()





