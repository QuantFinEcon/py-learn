import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#==============================================================================
# get data
#==============================================================================
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X.shape
np.unique(y)

#==============================================================================
# CROSS VALIDATION
#==============================================================================
from sklearn.cross_validation import train_test_split

#randomly split 30% test, 70% train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#==============================================================================
# PREPROCESSING - standardise/transform/linearise scale
#==============================================================================
from sklearn.preprocessing import StandardScaler

sc = StandardScaler() # initialise
sc.fit(X_train) # estimate mean, stdev
# transform uses those 'fit' estimated parameters for both train and test
# so they are comparable to each other
X_train_std, X_test_std = sc.transform(X_train), sc.transform(X_test) 

#==============================================================================
# perceptron linear model
#==============================================================================

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0) #initialise object
ppn.fit(X_train_std, y_train) #fit on train
y_pred = ppn.predict(X_test_std) #predict

print('Misclassified samples(%%): %3.2f %%' 
      % (float((y_test != y_pred).sum())/y_test.shape[0]*100.0) )

from sklearn.metrics import accuracy_score
print('Accuracy: %.4f%%:' % accuracy_score(y_test,y_pred))

#==============================================================================
# plot linear boundaries
#==============================================================================
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
  # setup marker generator and color map
  markers = ('s', 'x', 'o', '^', 'v')
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
  #any classifier with method predict
  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())
  # plot class samples
  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)

  # highlight test samples
  if test_idx:
    X_test, y_test = X[test_idx, :], y[test_idx]
    plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='test set')


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, 
                      classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()




#==============================================================================
# #  LOGISTIC REGRESSION
#==============================================================================
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
sigmoid = lambda z: 1./(1. + np.exp(-z))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.ylabel('$\phi (z)$')

from sklearn.linear_model import LogisticRegression

# C lower, more regularised, stable coefficients
lr = LogisticRegression(C=1000., random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined, classifier=lr,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

lr.predict_proba(X_test_std)


#==============================================================================
# REGULARISATION COEFFICIENT THAT PENEALISE EXTREME COEFFICIENTS
# lambda = 1/ C
#==============================================================================

weights, params = [], []
for c in np.arange(-5, 5):
  # smaller C, more stable Xi weight coeff
  lr = LogisticRegression(C=10**c, random_state=0)
  lr.fit(X_train_std, y_train)
  weights.append(lr.coef_[1])
  params.append(10**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()


#==============================================================================
# SVM
#==============================================================================
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std,
                      y_combined, classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#==============================================================================
# Large data, insufficient memory, Stochastic gradient is solution
#==============================================================================
#sample variation causes unstable coefficients that's inconsistent for different samples
#partial fit supports online learning
#fast convergence to same output

#simply, choose loss function  (the only difference between diff methods)
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')


#==============================================================================
# kernelising SVM for non linear classification
#==============================================================================

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

#create nonlinear combinations of the original features to project them onto 
#a higher dimensional space via a mapping function where it becomes 
#linearly separable.

# kernel trick to be computationally inexpensive is to replace dot product
# with kernel functions before dot product

# RADICAL BASIS FUNCTION / GAUSSIAN KERNEL - btw 0(dissimilar)-1(similar)
# free parameters (gamma, C) can be optimised as hyper parameters
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
#gamma = cutoff for gaussian sphere
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

# higher gamma => softer decision (more flexible) boundary
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# too high gamma => expect high generalisation error on unseen data
svm = SVC(kernel='rbf', random_state=0, gamma=100, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


#==============================================================================
# DECISION TREE
#==============================================================================

#impurity measures - gini and Entropy captures change in class probabilities
# gini and entropy very similar, time better spent on optimising pruning
import matplotlib.pyplot as plt
import numpy as np

def gini(p): return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p): return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p): return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
  ['Entropy', 'Entropy (scaled)',
  'Gini Impurity',
  'Misclassification Error'],
  ['-', '-', '--', '-.'],
  ['black', 'lightgray',
  'red', 'green', 'cyan']):
  line = ax.plot(x, i, label=lab,
  linestyle=ls, lw=2, color=c)
  
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
  ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105,150))
plt.legend(loc='upper left')

export_graphviz(tree,out_file='tree.dot',
                feature_names=['petal length', 'petal width'])
# install GraphViz
# cmd > dot -Tpng tree.dot -o tree.png

#==============================================================================
# RANDOM FOREST
#==============================================================================
#ensemble learning is to combine weak learners  to build a more robust model, 
# a strong learner. Less susceptible to overfitting and high generalised error

from sklearn.ensemble import RandomForestClassifier

# n_estimators=10  =>  10 trees in forest
# n_job = 2  =>  2 parallel core for fit and predict
forest = RandomForestClassifier(criterion='entropy',n_estimators=10,
                                random_state=1,n_jobs=2)
RandomForestClassifier()
forest.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105,150))
plt.legend(loc='upper left')














