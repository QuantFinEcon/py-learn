import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Perceptron(object):
  import numpy as np
  #  Perceptron classifier.

  def __init__(self, eta=0.01, n_iter=10):
    self.eta = float(eta) # between 0-1
    self.n_iter = int(n_iter)
  
  def fit(self, X, y):
    # Fit training data.
    # class attributes initialised on calling object methods self.w_
    self.w_ = np.zeros(1 + X.shape[1]) # initialise weights, +1 intercept
    self.errors_ = [] 
    for _ in range(self.n_iter):
      errors = 0
      for xi, target in zip(X, y): #truncated to shortest sample size of X,y
        """perceptron learning rule """
        update = self.eta * (target - self.predict(xi)) # no change if predicts correctly
        self.w_[1:] += update * xi # update weight to converge by learning rate to optimum
        self.w_[0] += update # update tol threshold
        errors += int(update != 0.0) # collect count (T or 1) of misclassifications
      self.errors_.append(errors) # error for that iteration
    return self
  
  def net_input(self, X):
    # net input
    return np.dot(X, self.w_[1:]) + self.w_[0] # dot.product
  
  def predict(self, X):
    # heaveside step fn
    return np.where(self.net_input(X) >= 0.0, 1, -1)

#==============================================================================
# get data
#==============================================================================

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()
df.info()

y=df.iloc[0:100,4].values # flatten df to array
y=np.where(y=='Iris-setosa',-1,1) # quantify output.. can extend to one vs rest for >=2 class
X = df.iloc[0:100, [0,2]].values
plt.scatter(X[:50,0],X[:50,1], color='red', marker='o', label='versicolor')
plt.scatter(X[50:100,0],X[50:100,1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

#==============================================================================
# error convergence
#==============================================================================
ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_/X.shape[0], marker='o')
plt.xlabel('error')
plt.ylabel('Number of misclassifications')
plt.title('Convergence of misclassification error')
plt.show()

#==============================================================================
# decision boundary
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


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

#==============================================================================
# ADAptive LInear NEuron classifier
#==============================================================================
class AdalineGD(object):
  
  def __init__(self, eta=0.01, n_iter=50):
    self.eta = eta
    self.n_iter = n_iter
  
  def fit(self, X, y):
    self.w_ = np.zeros(1 + X.shape[1])
    self.cost_ = []

    for i in range(self.n_iter):
      output = self.net_input(X)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = (errors**2).sum() / 2.0
      self.cost_.append(cost)
    return self
  
  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]
  
  def activation(self, X):
    """Compute linear activation"""
    return self.net_input(X)
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(X) >= 0.0, 1, -1)


#==============================================================================
# try out different model parameters (learning rate)
#==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4)) # 2 plots
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('error')
ax[0].set_ylabel('log(Sum-squared-error)') # cost reduction, too huge per jump i.e. log
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

# feature transformation: standardisation for convergence to convex error
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

#==============================================================================
# if need sampling, but introduce sampling variation, if batch for big data inefficient
#==============================================================================

from numpy.random import seed

class AdalineSGD(object):

  """
  shuffle : bool (default: True)
    Shuffles training data every epoch
    if True to prevent cycles.
  random_state : int (default: None)
    Set random state for shuffling
    and initializing the weights.
  """
  
  def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
    self.eta = eta
    self.n_iter = n_iter
    self.w_initialized = False
    self.shuffle = shuffle
    if random_state:
      seed(random_state)
  
  def fit(self, X, y):
    self._initialize_weights(X.shape[1])
    self.cost_ = []
    for i in range(self.n_iter):
      if self.shuffle: X, y = self._shuffle(X, y)
      cost = []
      for xi, target in zip(X, y): 
        cost.append(self._update_weights(xi, target)) # no compressing to +-1
      avg_cost = sum(cost)/len(y)
      self.cost_.append(avg_cost)
    return self
  
  def partial_fit(self, X, y):
    """Fit training data without reinitializing the weights"""
    if not self.w_initialized:
      self._initialize_weights(X.shape[1]) # reuse old sample coefficients
    if y.ravel().shape[0] > 1:
      for xi, target in zip(X, y):
        self._update_weights(xi, target)
    else:
      self._update_weights(X, y)
    return self
  
  def _shuffle(self, X, y):
    """Shuffle training data"""
    r = np.random.permutation(len(y))
    return X[r], y[r]
  
  def _initialize_weights(self, m):
    """Initialize weights to zeros"""
    self.w_ = np.zeros(1 + m)
    self.w_initialized = True
  
  def _update_weights(self, xi, target):
    """Apply Adaline adaptive learning rule to update the weights"""
    output = self.net_input(xi)
    error = (target - output)
    self.w_[1:] += self.eta * xi.dot(error)
    self.w_[0] += self.eta * error
    cost = 0.5 * error**2
    return cost
  
  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]
  
  def activation(self, X):
    """Compute linear activation"""
    return self.net_input(X)
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(X) >= 0.0, 1, -1)

#==============================================================================
# Stochastic DG converges faster
#==============================================================================
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()






