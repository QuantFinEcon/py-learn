
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

boston = load_boston(); print(boston.keys())
iris = load_iris(); print(iris.keys())
digits = load_digits(); print(digits.keys())
diabetes = load_diabetes(); print(diabetes.keys())
wine = load_wine(); print(wine.keys())
breast_cancer = load_breast_cancer(); print(breast_cancer.keys())
faces = fetch_olivetti_faces() #downloading to os.getcwd()

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

from sklearn.manifold import (Isomap,
                              LocallyLinearEmbedding,
                              MDS,
                              SpectralEmbedding,
                              TSNE,)

from sklearn.decomposition import (PCA,
                                   IncrementalPCA,
                                   FactorAnalysis,
                                   FastICA,
                                   KernelPCA,
                                   DictionaryLearning,
                                   LatentDirichletAllocation,
                                   MiniBatchDictionaryLearning,
                                   MiniBatchSparsePCA,
                                   NMF,
                                   SparseCoder,
                                   SparsePCA,
                                   TruncatedSVD,)


"""
###############################################################################
http://scikit-learn.org/stable/modules/manifold.html#manifold
http://scikit-learn.org/stable/modules/decomposition.html#decompositions

Manifold
- non-linear dimensionality reduction
- unsupervised => use data to project itself
- idea that the dimensionality of many data sets is only artificially high
- High-dimensional datasets can be very difficult to visualize and unintuitive
- generalize linear frameworks like PCA to be 
    sensitive to non-linear structure in data
Versus linear projections
- Principal Component Analysis (PCA)
- Independent Component Analysis
- Linear Discriminant Analysis



###############################################################################
"""



"""
Isomap?
- Non-linear dimensionality reduction through Isometric Mapping
- an extension of Multi-dimensional Scaling (MDS) or Kernel PCA
- maintains geodesic distances between all points

Parameters
----------
n_neighbors : integer
    number of neighbors to consider for each point.

n_components : integer
    number of coordinates for the manifold

eigen_solver : ['auto'|'arpack'|'dense']
    'auto' : Attempt to choose the most efficient solver
    for the given problem.

    'arpack' : Use Arnoldi decomposition to find the eigenvalues
    and eigenvectors.

    'dense' : Use a direct solver (i.e. LAPACK)
    for the eigenvalue decomposition.

tol : float
    Convergence tolerance passed to arpack or lobpcg.
    not used if eigen_solver == 'dense'.

max_iter : integer
    Maximum number of iterations for the arpack solver.
    not used if eigen_solver == 'dense'.

path_method : string ['auto'|'FW'|'D']
    Method to use in finding shortest path.

    'auto' : attempt to choose the best algorithm automatically.

    'FW' : Floyd-Warshall algorithm.

    'D' : Dijkstra's algorithm.

neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
    Algorithm to use for nearest neighbors search,
    passed to neighbors.NearestNeighbors instance.

n_jobs : int, optional (default = 1)
    The number of parallel jobs to run.
    If ``-1``, then the number of jobs is set to the number of CPU cores.

Attributes
----------
embedding_ : array-like, shape (n_samples, n_components)
    Stores the embedding vectors.

kernel_pca_ : object
    `KernelPCA` object used to implement the embedding.

training_data_ : array-like, shape (n_samples, n_features)
    Stores the training data.

nbrs_ : sklearn.neighbors.NearestNeighbors instance
    Stores nearest neighbors instance, including BallTree or KDtree
    if applicable.

dist_matrix_ : array-like, shape (n_samples, n_samples)
    Stores the geodesic distance matrix of training data.
"""

y=iris.target
X=iris.data
X = StandardScaler().fit_transform(X)

isomap = Isomap(n_neighbors=10, n_components=2, neighbors_algorithm='auto')
X_ = isomap.fit_transform(X)
isomap.dist_matrix_
isomap.dist_matrix_.shape
isomap.nbrs_
isomap.embedding_
isomap.kernel_pca_

plt.scatter(X_[:,0], X_[:,1], c=iris.target)

isomap = Isomap(n_neighbors=2, n_components=2, neighbors_algorithm='auto')
X_ = isomap.fit_transform(X)
plt.scatter(X_[:,0], X_[:,1], c=iris.target)

isomap = Isomap(n_neighbors=5, n_components=2, neighbors_algorithm='auto')
X_ = isomap.fit_transform(X)
plt.scatter(X_[:,0], X_[:,1], c=iris.target)

y=wine.target
X=wine.data
X.shape
X = StandardScaler().fit_transform(X)

isomap = Isomap(n_neighbors=5, n_components=2, neighbors_algorithm='auto')
X_ = isomap.fit_transform(X)
plt.scatter(X_[:,0], X_[:,1], c = wine.target)


"""
PCA
Linear dimensionality reduction using Singular Value Decomposition of the
data to project it to a lower dimensional space.

uses the LAPACK implementation of the full SVD or a randomized truncated SVD
scipy.sparse.linalg ARPACK implementation of the truncated SVD.

- important to scale and standardise variance to 1.0 and mean to 0
    preprocessing.StandardScaler()
- rank components contribution to total variance
- 1st component explains most variance
- components are linearly independent => correlation=0

PCA?
Init signature: PCA(n_components=None, copy=True, whiten=False, 
                    svd_solver='auto', tol=0.0, iterated_power='auto', 
                    random_state=None)
Parameters
----------
n_components : int, float, None or string
    Number of components to keep.
    if n_components is not set all components are kept::

        n_components == min(n_samples, n_features)

    if n_components == 'mle' and svd_solver == 'full', Minka's MLE is used
    to guess the dimension
    if ``0 < n_components < 1`` and svd_solver == 'full', select the number
    of components such that the amount of variance that needs to be
    explained is greater than the percentage specified by n_components
    n_components cannot be equal to n_features for svd_solver == 'arpack'.

copy : bool (default True)
    If False, data passed to fit are overwritten and running
    fit(X).transform(X) will not yield the expected results,
    use fit_transform(X) instead.

whiten : bool, optional (default False)
    When True (False by default) the `components_` vectors are multiplied
    by the square root of n_samples and then divided by the singular values
    to ensure uncorrelated outputs with unit component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometime
    improve the predictive accuracy of the downstream estimators by
    making their data respect some hard-wired assumptions.

svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
    auto :
        the solver is selected by a default policy based on `X.shape` and
        `n_components`: if the input data is larger than 500x500 and the
        number of components to extract is lower than 80% of the smallest
        dimension of the data, then the more efficient 'randomized'
        method is enabled. Otherwise the exact full SVD is computed and
        optionally truncated afterwards.
    full :
        run exact full SVD calling the standard LAPACK solver via
        `scipy.linalg.svd` and select the components by postprocessing
    arpack :
        run SVD truncated to n_components calling ARPACK solver via
        `scipy.sparse.linalg.svds`. It requires strictly
        0 < n_components < X.shape[1]
    randomized :
        run randomized SVD by the method of Halko et al.

    .. versionadded:: 0.18.0

tol : float >= 0, optional (default .0)
    Tolerance for singular values computed by svd_solver == 'arpack'.

    .. versionadded:: 0.18.0

iterated_power : int >= 0, or 'auto', (default 'auto')
    Number of iterations for the power method computed by
    svd_solver == 'randomized'.

    .. versionadded:: 0.18.0

random_state : int, RandomState instance or None, optional (default None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    .. versionadded:: 0.18.0

Attributes
----------
components_ : array, shape (n_components, n_features)
    Principal axes in feature space, representing the directions of
    maximum variance in the data. The components are sorted by
    ``explained_variance_``.

explained_variance_ : array, shape (n_components,)
    The amount of variance explained by each of the selected components.

    Equal to n_components largest eigenvalues
    of the covariance matrix of X.

    .. versionadded:: 0.18

explained_variance_ratio_ : array, shape (n_components,)
    Percentage of variance explained by each of the selected components.

    If ``n_components`` is not set then all components are stored and the
    sum of explained variances is equal to 1.0.

singular_values_ : array, shape (n_components,)
    The singular values corresponding to each of the selected components.
    The singular values are equal to the 2-norms of the ``n_components``
    variables in the lower-dimensional space.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.

    Equal to `X.mean(axis=0)`.

n_components_ : int
    The estimated number of components. When n_components is set
    to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
    number is estimated from input data. Otherwise it equals the parameter
    n_components, or n_features if n_components is None.

noise_variance_ : float
    The estimated noise covariance following the Probabilistic PCA model
    from Tipping and Bishop 1999. See "Pattern Recognition and
    Machine Learning" by C. Bishop, 12.2.1 p. 574 or
    http://www.miketipping.com/papers/met-mppca.pdf. It is required to
    computed the estimated data covariance and score samples.

    Equal to the average of (min(n_features, n_samples) - n_components)
    smallest eigenvalues of the covariance matrix of X.
"""

def pca_biplot(X, coeff, components=[1,2], labels=None):
    """
    @X: standardised data used in PCA
    @coeff: pca.components_
    @components: list, [1,2] PC1 against PC2
    @labels: intuitive description of component given coefficients of components
    """
    X = X[:,[x-1 for x in components]]
    coeff = coeff[[x-1 for x in components],:]
    xs = X[:,0]
    ys = X[:,1]
    coeff = np.transpose(coeff)
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())

    #biplot
    plt.figure(figsize=(10,10))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(components[0]))
    plt.ylabel("PC{}".format(components[1]))
    plt.grid()
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), 
                     color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], 
                     color = 'g', ha = 'center', va = 'center')
    plt.show()

def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)

from scipy import stats

y=iris.target
X=iris.data
stats.describe(X) # uneven variance
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
X_ = pca.fit_transform(X)
pca.components_
pca.explained_variance_
pca.explained_variance_ratio_

pca_biplot(X_, pca.components_, (1,2), labels=iris.feature_names)
pca_biplot(X_, pca.components_, (1,3), labels=iris.feature_names)
pca_biplot(X_, pca.components_, (2,3), labels=iris.feature_names)

# varimax rotation of components
varimax(pca.components_).round(2)
iris.feature_names

"""
IncrementalPCA?
Init signature: IncrementalPCA(n_components=None, whiten=False, 
                               copy=True, batch_size=None)

a replacement for principal component analysis (PCA) 
when the dataset to be decomposed is too large to fit in memory. 
IPCA builds a low-rank approximation for the input data using 
an amount of memory which is independent of the number of input data samples. 
It is still dependent on the input data features, 
but changing the batch size allows for control of memory usage.

The computational overhead of each SVD is
``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
remain in memory at a time. 

SparsePCA?
Init signature: SparsePCA(n_components=None, alpha=1, ridge_alpha=0.01, 
                          max_iter=1000, tol=1e-08, method='lars', n_jobs=1, 
                          U_init=None, V_init=None, verbose=False, random_state=None)
Finds the set of sparse components that can optimally reconstruct
the data.  The amount of sparseness is controllable by the coefficient
of the L1 penalty, given by the parameter alpha.

TruncatedSVD?
TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, 
             random_state=None, tol=0.0)
SVD suffers from a problem called "sign indeterminancy", which means the
sign of the ``components_`` and the output from transform depend on the
algorithm and random state. To work around this, fit instances of this
class to data once, then keep the instance around to do transformations.

"""
y=iris.target
X=iris.data
X.shape
stats.describe(X) # uneven variance
X = StandardScaler().fit_transform(X)
pca = IncrementalPCA(n_components=3, batch_size=50)
X_ = pca.fit_transform(X)
pca.components_
pca.explained_variance_
pca.explained_variance_ratio_

pca_biplot(X_, pca.components_, (1,2), labels=iris.feature_names)
pca_biplot(X_, pca.components_, (1,3), labels=iris.feature_names)
pca_biplot(X_, pca.components_, (2,3), labels=iris.feature_names)


"""
http://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds

Multidimensional Scaling (MDS)
-  a low-dimensional representation of the data in which the distances 
    respect well the relative distances in the original high-dimensional space.
- model similarity or dissimilarity data as distances in a geometric spaces
- OPEN TO CUSTOM MEASUREMENT OF DISSIMILARITY BETWEEN ROWS OF N-FEATURES

- two types of MDS algorithm: metric and non metric
    
metric - similarity matrix arises from a metric (and thus respects the 
         triangular inequality), the distances between output two points are 
         then set to be as close as possible to the similarity or dissimilarity data
         Know as 'absolute' MDS
         
non-metric - algorithms will try to preserve the order of the distances, 
            and hence seek for a monotonic relationship between the distances 
            in the embedded space and the similarities/dissimilarities

obj to min is stress
stress = foreach(i<j)sum [dist(i,j)(X) - transformed_dist(i,j)(X)]

Init signature: MDS(n_components=2, metric=True, n_init=4, max_iter=300, 
                    verbose=0, eps=0.001, n_jobs=1, random_state=None, 
                    dissimilarity='euclidean')

Parameters
----------
n_components : int, optional, default: 2
    Number of dimensions in which to immerse the dissimilarities.

metric : boolean, optional, default: True
    If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.

n_init : int, optional, default: 4
    Number of times the SMACOF algorithm will be run with different
    initializations. The final results will be the best output of the runs,
    determined by the run with the smallest final stress.

max_iter : int, optional, default: 300
    Maximum number of iterations of the SMACOF algorithm for a single run.

verbose : int, optional, default: 0
    Level of verbosity.

eps : float, optional, default: 1e-3
    Relative tolerance with respect to stress at which to declare
    convergence.

n_jobs : int, optional, default: 1
    The number of jobs to use for the computation. If multiple
    initializations are used (``n_init``), each run of the algorithm is
    computed in parallel.

    If -1 all CPUs are used. If 1 is given, no parallel computing code is
    used at all, which is useful for debugging. For ``n_jobs`` below -1,
    (``n_cpus + 1 + n_jobs``) are used. Thus for ``n_jobs = -2``, all CPUs
    but one are used.

random_state : int, RandomState instance or None, optional, default: None
    The generator used to initialize the centers.  If int, random_state is
    the seed used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number
    generator is the RandomState instance used by `np.random`.

dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
    Dissimilarity measure to use:

    - 'euclidean':
        Pairwise Euclidean distances between points in the dataset.

    - 'precomputed':
        Pre-computed dissimilarities are passed directly to ``fit`` and
        ``fit_transform``.

Attributes
----------
embedding_ : array-like, shape (n_components, n_samples)
    Stores the position of the dataset in the embedding space.

stress_ : float
    The final value of the stress (sum of squared distance of the
    disparities and the distances for all constrained points).
    
"""

y=iris.target
X=iris.data
X = StandardScaler().fit_transform(X)

mds = MDS(n_components=2, metric=True, dissimilarity='euclidean', n_init=20,
          n_jobs=1, verbose=True)
X_ = mds.fit_transform(X)
mds.embedding_
mds.stress_
mds.dissimilarity_matrix_
mds.dissimilarity_matrix_.shape

plt.scatter(X_[:,0], X_[:,1], c=iris.target)
plt.axvline(0);plt.axhline(0)

#non-metric MDS
nmds = MDS(n_components=2, metric=False, dissimilarity='euclidean', n_init=1,
           n_jobs=1, verbose=True)
nX_ = nmds.fit_transform(X, init=mds.embedding_)
plt.scatter(nX_[:,0], nX_[:,1], c=iris.target)
plt.axvline(0);plt.axhline(0)

y=wine.target
X=wine.data
X.shape
X = StandardScaler().fit_transform(X)

mds = MDS(n_components=2, metric=True, dissimilarity='euclidean', n_init=20,
          n_jobs=1, verbose=True)
X_ = mds.fit_transform(X)
mds.embedding_
mds.stress_
mds.dissimilarity_matrix_
mds.dissimilarity_matrix_.shape
plt.scatter(X_[:,0], X_[:,1], c=wine.target)
plt.axvline(0);plt.axhline(0)

mds = MDS(n_components=3, metric=True, dissimilarity='euclidean', n_init=20,
          n_jobs=1, verbose=True)
X_ = mds.fit_transform(X)
mds.embedding_
mds.stress_
mds.dissimilarity_matrix_
mds.dissimilarity_matrix_.shape
plt.scatter(X_[:,0], X_[:,1], c=wine.target)
plt.scatter(X_[:,1], X_[:,2], c=wine.target)
plt.scatter(X_[:,0], X_[:,2], c=wine.target)


"""
Kernal PCA
- choice of kernel to best find projection in data 
    that's not linearly seperable

http://scikit-learn.org/stable/modules/metrics.html#metrics
sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS
sklearn.metrics.pairwise.PAIRWISE_BOOLEAN_FUNCTIONS
sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS

from sklearn.metrics.pairwise import (linear_kernel,
                                      cosine_similarity, 
                                      #L2 normalised equivalent to linear, 
                                      #euclidean measure distance, cos measure angle
                                      polynomial_kernel, # degree d
                                      sigmoid_kernel, #tanh, NN activation function
                                      rbf_kernel, # exp(|X-Y|^2)
                                      laplacian_kernel, # exp(|X-Y|)
                                      chi2_kernel, 
                                      # data non-negative, normalised to (0,1) like discrete
                                      )

"""
X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kpca = kpca.fit_transform(X)
#X_back = kpca.inverse_transform(X_kpca)
#X_back == X

plt.figure()
plt.subplot(1, 2, 1, aspect='equal')
plt.scatter(X[:, 0], X[:,1], c=y, s=20, edgecolor='k')
plt.title("Original space")
plt.xlabel("$x^1$")
plt.ylabel("$x_2$")

plt.subplot(1, 2, 2, aspect='equal')
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")


"""
Factor Analysis

linear generative model with Gaussian latent variables.

- observations are assumed to be caused by a linear transformation of
    lower dimensional latent factors and unequal added Gaussian noise.
- Factors are distributed according to a Gaussian with zero mean and unit covariance. 
- The noise is also zero mean and has an arbitrary diagonal covariance matrix.

If we would restrict the model further, by assuming that the Gaussian
noise is even isotropic (all diagonal entries are the same) we would obtain
PCA

http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
compare PCA and FA with cross-validation on low rank data 
corrupted with homoscedastic noise (noise variance is the same for each feature) 
or heteroscedastic noise (noise variance is the different for each feature).
PCA fails and overestimates the rank when heteroscedastic noise is present. 

Parameters
----------
n_components : int | None
    Dimensionality of latent space, the number of components
    of ``X`` that are obtained after ``transform``.
    If None, n_components is set to the number of features.

tol : float
    Stopping tolerance for EM algorithm.

copy : bool
    Whether to make a copy of X. If ``False``, the input X gets overwritten
    during fitting.

max_iter : int
    Maximum number of iterations.

noise_variance_init : None | array, shape=(n_features,)
    The initial guess of the noise variance for each feature.
    If None, it defaults to np.ones(n_features)

svd_method : {'lapack', 'randomized'}
    Which SVD method to use. If 'lapack' use standard SVD from
    scipy.linalg, if 'randomized' use fast ``randomized_svd`` function.
    Defaults to 'randomized'. For most applications 'randomized' will
    be sufficiently precise while providing significant speed gains.
    Accuracy can also be improved by setting higher values for
    `iterated_power`. If this is not sufficient, for maximum precision
    you should choose 'lapack'.

iterated_power : int, optional
    Number of iterations for the power method. 3 by default. Only used
    if ``svd_method`` equals 'randomized'

random_state : int, RandomState instance or None, optional (default=0)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Only used when ``svd_method`` equals 'randomized'.

Attributes
----------
components_ : array, [n_components, n_features]
    Components with maximum variance.

loglike_ : list, [n_iterations]
    The log likelihood at each iteration.

noise_variance_ : array, shape=(n_features,)
    The estimated noise variance for each feature.

n_iter_ : int
    Number of iterations run.
"""

y=iris.target
X=iris.data
X = StandardScaler().fit_transform(X)

fa = FactorAnalysis(n_components=2,
                    noise_variance_init=np.ones(X.shape[1]),
                    svd_method = 'lapack')
X_ = fa.fit_transform(X)
fa.components_
fa.loglike_
fa.noise_variance_

plt.scatter(X_[:,0], X_[:,1], c=iris.target); plt.show()
pca_biplot(X_, fa.components_, (1,2), labels=iris.feature_names)


y=wine.target
X=wine.data
X = StandardScaler().fit_transform(X)

fa = FactorAnalysis(n_components=2,
                    noise_variance_init=np.ones(X.shape[1]),
                    svd_method = 'lapack')
X_ = fa.fit_transform(X)
fa.components_
fa.loglike_
fa.noise_variance_

plt.scatter(X_[:,0], X_[:,1], c=wine.target); plt.show()
pca_biplot(X_, fa.components_, (1,2), labels=wine.feature_names)


y=breast_cancer.target
X=breast_cancer.data
X = StandardScaler().fit_transform(X)

fa = FactorAnalysis(n_components=2,
                    noise_variance_init=np.ones(X.shape[1]),
                    svd_method = 'lapack')
X_ = fa.fit_transform(X)
fa.components_
fa.loglike_
fa.noise_variance_

plt.scatter(X_[:,0], X_[:,1], c=breast_cancer.target); plt.show()
pca_biplot(X_, fa.components_, (1,2), labels=breast_cancer.feature_names)


"""
Independent Component Analysis (ICA)
- separates a multivariate signal into additive subcomponents that are maximally independent
- ICA model does not include a noise term, for the model to be correct, whitening must be applied
- ICA is not used for reducing dimensionality but for separating superimposed signals
- separate mixed signals (a problem known as blind source separation)

ICA is an algorithm that finds directions in the feature space corresponding 
to projections with high non-Gaussianity. PCA, on the other hand, finds 
orthogonal directions in the raw feature space that correspond to 
directions accounting for maximum variance. 

PCA decorrelates outputs, ICA make outputs statistically independent
with an unmixing matrix to retrieve source, although
it cannot bring back orderings and identity of sources

FastICA?


"""
from scipy import signal

# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise
S /= S.std(axis=0)  # whitening

# Mix data
A = np.array([[1, -0.8, 0.9], 
              [-0.8, 1, 0.6], 
              [0.9, 0.6, 1]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
# Returns True if two arrays are element-wise equal within a tolerance.
np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# Plot results
plt.figure(figsize=(20,20))

models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()




