
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
                              make_blobs)

from sklearn.cluster import (KMeans,
                             MiniBatchKMeans,
                             DBSCAN,
                             FeatureAgglomeration,
                             AgglomerativeClustering,
                             AffinityPropagation,
                             MeanShift,
                             estimate_bandwidth,

Birch,
SpectralBiclustering,
SpectralClustering,
SpectralCoclustering,
affinity_propagation,
affinity_propagation_,
bicluster,
estimate_bandwidth,
get_bin_seeds,
hierarchical?,
linkage_tree?,
ward_tree?)

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
                                   PCA,
                                   RandomizedPCA,
                                   SparseCoder,
                                   SparsePCA,
                                   TruncatedSVD,)

"""
K-Means needs implicit assumption around data, otherwise, leads to 
underdesirable clusters
- incorrect k
- unequal variance
- Anisotropy distribution
    property of being directionally dependent, which implies different 
    properties in different directions, as opposed to isotropy.
    Isotropy is uniformity in all orientations
- unevenly distributed clusters (majority vs minority, so long segregated it works)

make_blobs?
Signature: make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, 
                      center_box=(-10.0, 10.0), shuffle=True, random_state=None)
Docstring:
Generate isotropic Gaussian blobs for clustering.

KMeans?
Init signature: KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, 
                       tol=0.0001, precompute_distances='auto', verbose=0, 
                       random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

Parameters
----------

n_clusters : int, optional, default: 8
    The number of clusters to form as well as the number of
    centroids to generate.

init : {'k-means++', 'random' or an ndarray}
    Method for initialization, defaults to 'k-means++':

    'k-means++' : selects initial cluster centers for k-mean
    clustering in a smart way to speed up convergence. See section
    Notes in k_init for more details.

    'random': choose k observations (rows) at random from data for
    the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features)
    and gives the initial centers.

n_init : int, default: 10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.

max_iter : int, default: 300
    Maximum number of iterations of the k-means algorithm for a
    single run.

tol : float, default: 1e-4
    Relative tolerance with regards to inertia to declare convergence

precompute_distances : {'auto', True, False}
    Precompute distances (faster but takes more memory).

    'auto' : do not precompute distances if n_samples * n_clusters > 12
    million. This corresponds to about 100MB overhead per job using
    double precision.

    True : always precompute distances

    False : never precompute distances

algorithm : "auto", "full" or "elkan", default="auto"
    K-means algorithm to use. The classical EM-style algorithm is "full".
    The "elkan" variation is more efficient by using the triangle
    inequality, but currently doesn't support sparse data. "auto" chooses
    "elkan" for dense data and "full" for sparse data.

"""
plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=3)

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, # <========
                random_state=random_state).fit_predict(X)

plt.subplot(221) # 221 means 2x2 grid 1st plot ==> plt.subplot(2,2,1)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]# <========
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5], # <========
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,
                random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()

X,y = make_blobs(n_samples=3000, centers=5, n_features=2,
                 cluster_std=[1,1,1,1,1])
kmeans = KMeans(n_clusters=5).fit(X)

pca  = PCA(n_components = len(set(y)) ).fit(X)
pca.components_
pca.explained_variance_
pca.explained_variance_ratio_
kmeans = KMeans(n_clusters=5, init=pca.components_).fit(X)

kmeans.labels_
kmeans.cluster_centers_
plt.scatter(X[:,0],X[:,1],c=kmeans.fit_predict(X))
plt.scatter(X[:,0],X[:,1],c=y) # real label

"""
MiniBatchKMeans?
Init signature: MiniBatchKMeans(n_clusters=8, init='k-means++', max_iter=100, 
                                batch_size=100, verbose=0, compute_labels=True, 
                                random_state=None, tol=0.0, max_no_improvement=10, 
                                init_size=None, n_init=3, reassignment_ratio=0.01)
MiniBatchKMeans
    Alternative online implementation that does incremental updates
    of the centers positions using mini-batches.
    For large scale learning (say n_samples > 10k) MiniBatchKMeans is
    probably much faster than the default batch implementation.

Parameters
----------

batch_size : int, optional, default: 100
    Size of the mini batches.

tol : float, default: 0.0
    Control early stopping based on the relative center changes as
    measured by a smoothed, variance-normalized of the mean center
    squared position changes. This early stopping heuristics is
    closer to the one used for the batch variant of the algorithms
    but induces a slight computational and memory overhead over the
    inertia heuristic.

    To disable convergence detection based on normalized center
    change, set tol to 0.0 (default).

max_no_improvement : int, default: 10
    Control early stopping based on the consecutive number of mini
    batches that does not yield an improvement on the smoothed inertia.

    To disable convergence detection based on inertia, set
    max_no_improvement to None.

init_size : int, optional, default: 3 * batch_size
    Number of samples to randomly sample for speeding up the
    initialization (sometimes at the expense of accuracy): the
    only algorithm is initialized by running a batch KMeans on a
    random subset of the data. This needs to be larger than n_clusters.

n_init : int, default=3
    Number of random initializations that are tried.
    In contrast to KMeans, the algorithm is only run once, using the
    best of the ``n_init`` initializations as measured by inertia.

reassignment_ratio : float, default: 0.01
    Control the fraction of the maximum number of counts for a
    center to be reassigned. A higher value means that low count
    centers are more easily reassigned, which means that the
    model will take longer to converge, but should converge in a
    better clustering.
    
"""
X,y = make_blobs(n_samples=3000, centers=5, n_features=2,
                 cluster_std=[1,1,1,1,1], random_state=1)
kmeans = MiniBatchKMeans(n_clusters=5, batch_size=300, 
                         max_iter=500, max_no_improvement=100,
                         verbose=1).fit(X)

kmeans.labels_
kmeans.cluster_centers_

plt.figure(figsize=(24,12))
plt.subplot(121)
plt.scatter(X[:,0],X[:,1],c=kmeans.fit_predict(X))
plt.subplot(122)
plt.scatter(X[:,0],X[:,1],c=y) # real label
plt.show()

"""
AffinityPropagation?
Similar to DBSCAN. Group points based on proximity. 
it chooses the number of clusters based on the data provided. 
suitable for small - mid sized data

two important parameters are the preference, which controls how 
many exemplars are used, and the damping factor which damps the 
responsibility and availability messages to 
avoid numerical oscillations when updating these messages.

Drawback: high time complexity n_samples^2 * n_iter
            high memory complexity, esp for dense matrix

signature: AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, 
                               copy=True, preference=None, affinity='euclidean', 
                               verbose=False)

Parameters
----------
damping : float, optional, default: 0.5
    Damping factor (between 0.5 and 1) is the extent to
    which the current value is maintained relative to
    incoming values (weighted 1 - damping). This in order
    to avoid numerical oscillations when updating these
    values (messages).

max_iter : int, optional, default: 200
    Maximum number of iterations.

convergence_iter : int, optional, default: 15
    Number of iterations with no change in the number
    of estimated clusters that stops the convergence.

copy : boolean, optional, default: True
    Make a copy of input data.

preference : array-like, shape (n_samples,) or float, optional
    Preferences for each point - points with larger values of
    preferences are more likely to be chosen as exemplars. The number
    of exemplars, ie of clusters, is influenced by the input
    preferences value. If the preferences are not passed as arguments,
    they will be set to the median of the input similarities.

affinity : string, optional, default=``euclidean``
    Which affinity to use. At the moment ``precomputed`` and
    ``euclidean`` are supported. ``euclidean`` uses the
    negative squared euclidean distance between points.

verbose : boolean, optional, default: False
    Whether to be verbose.


Attributes
----------
cluster_centers_indices_ : array, shape (n_clusters,)
    Indices of cluster centers

cluster_centers_ : array, shape (n_clusters, n_features)
    Cluster centers (if affinity != ``precomputed``).

labels_ : array, shape (n_samples,)
    Labels of each point

affinity_matrix_ : array, shape (n_samples, n_samples)
    Stores the affinity matrix used in ``fit``.

n_iter_ : int
    Number of iterations taken to converge.

"""










"""
MeanShift

- works well with tight clusters, small std dev

"""

# #############################################################################
# Generate sample data
X, y = make_blobs(n_samples=10000, centers=5, 
                  cluster_std=0.6, random_state=1)

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()









