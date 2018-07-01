
"""
Compared to train_test_split, cross-validation gives you 
a more reliable measure of your model's quality, though it 
takes longer to run.

choices about predictive variables to use, 
what types of models to use, 
what arguments to supply those models, etc. 

We make these choices in a data-driven way by 
measuring model quality of various alternatives.

The larger the test set, the less randomness (aka "noise") there is in our measure of model quality.

But we can only get a large test set by removing data 
from our training data, and smaller training datasets mean worse models. 
In fact, the ideal modeling decisions on a small dataset typically 
aren't the best modeling decisions on large datasets.

So, if your dataset is smaller, you should run cross-validation.

"""

import pandas as pd

data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import (check_cv, 
                                     cross_val_score,
                                     cross_val_predict,
                                     cross_validate)

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))

# =============================================================================
# CLASSIFICATION EVALUATION METRICS
# =============================================================================
"""
http://scikit-learn.org/stable/modules/model_evaluation.html

scoring:    
    higher return values are better than lower return values
    e.g. MAE as negative MAE
    different for: clustering / 
                    classification (binary or multiclass) / 
                    regression

from sklearn.metrics.classification import ...
from sklearn.metrics.cluster import ...
from sklearn.metrics.regression import ...

ending with _score return a value to maximize, the higher the better.

ending with _error or _loss return a value to minimize, the lower the better. 
converting into a scorer object using make_scorer, 
set the greater_is_better parameter to False 
"""

from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer)

import numpy as np
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

# loss_func will negate the return value of my_custom_loss_func,
#  which will be np.log(2), 0.693, given the values for ground_truth
#  and predictions defined below.

loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
score = make_scorer(my_custom_loss_func, greater_is_better=True)
ground_truth = [[1], [1]]
predictions  = [0, 1]
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf = clf.fit(ground_truth, predictions)
loss(clf,ground_truth, predictions) 
score(clf,ground_truth, predictions) 


"""
evaluation of multiple metrics in GridSearchCV, 
RandomizedSearchCV and cross_validate

only those scorer functions that return a single score can be 
passed inside the dict
functions that return multiple values are not permitted and 
will require a wrapper to return a single metric
"""
scoring = ['accuracy', 'precision']

from sklearn.metrics import accuracy_score
scoring = {'accuracy': make_scorer(accuracy_score),
           'prec': 'precision'}


"""
confusion matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
class_names = ['class 0', 'class 1', 'class 2']
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


"""
classification report
    precision: ability to avoid false positive
    recall: ability to find true positive
    F-measure: weighted harmonic mean of the precision and recall. 
                best value at 1 and its worst score at 0 

"""

from sklearn.metrics.classification import classification_report

target_names = ['class 0', 'class 1', 'class 2']
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
print(classification_report(y_true, y_pred, target_names=target_names))


"""
hamming distance:
    measures the minimum number of substitutions required to 
    change one string into the other, or 
    the minimum number of errors that could have 
    transformed one string into the other.    

log loss:
    logistic regression loss or cross-entropy loss, 
    is defined on probability estimates. 
    It is commonly used in (multinomial) logistic regression 
    and neural networks, as well as in some variants of 
    expectation-maximization, and can be used to evaluate 
    the probability outputs (predict_proba) of a classifier 
    instead of its discrete predictions
"""

from sklearn.metrics.classification import (hamming_loss,
                                            log_loss)

# binary class
y_pred = [1, 2, 3, 4]
y_true = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
y_true = [5, 6, 7, 8]
hamming_loss(y_true, y_pred)
hamming_loss(list("ABFD"), list("ABCD"))

#multi class
hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))

y_true = [0, 0, 1, 1]
y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]] # [Pr(0), Pr(1)]
log_loss(y_true, y_pred)

"""
Receiver operating characteristic (ROC) Curve

roc_curve?
roc_curve(y_true, y_score, pos_label=None, 
          sample_weight=None, drop_intermediate=True)
Note: this implementation is restricted to the binary classification task.

y_true : array, shape = [n_samples]
    True binary labels in range {0, 1} or {-1, 1}.  If labels are not
    binary, pos_label should be explicitly given.

y_score : array, shape = [n_samples]
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by "decision_function" on some classifiers).

pos_label : int or str, default=None
    Label considered as positive and others are considered negative.

sample_weight : array-like of shape = [n_samples], optional
    Sample weights.

"""

import numpy as np
from sklearn.metrics.ranking import (roc_curve, 
                                     auc,
                                     roc_auc_score)

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
fpr
tpr
thresholds


# =============================================================================
# REGRESSION EVALUATION METRICS
# =============================================================================

from sklearn.metrics.regression import (r2_score,
                                        explained_variance_score,
                                        mean_absolute_error,
                                        median_absolute_error,
                                        mean_squared_error,
                                        mean_squared_log_error)

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)
r2_score(y_true, y_pred)

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
mean_squared_error(y_true, y_pred)
r2_score(y_true, y_pred)


# =============================================================================
# CLUSTERING EVALUATION METRICS
# =============================================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
y[y != 1] = -1 #binary
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.dummy import (DummyClassifier, 
                           DummyRegressor)
from sklearn.svm import (SVC,
                         SVR,
                         LinearSVC,
                         LinearSVR)

"""
import sklearn.svm
dir(sklearn.svm)
 'classes',
 'l1_min_c',
 'liblinear',
 'libsvm',
 'libsvm_sparse'

import sklearn.svm.classes
dir(sklearn.svm.classes)
 'BaseEstimator',
 'BaseLibSVM',
 'BaseSVC',
 'LinearClassifierMixin',
 'LinearModel',
 'LinearSVC',
 'LinearSVR',
 'NuSVC',
 'NuSVR',
 'OneClassSVM',
 'RegressorMixin',
 'SVC',
 'SVR',
 'SparseCoefMixin'

SVC?
Init signature: SVC(C=1.0, kernel='rbf', degree=3, 
                    gamma='auto', coef0=0.0, shrinking=True, 
                    probability=False, tol=0.001, cache_size=200, 
                    class_weight=None, verbose=False, max_iter=-1, 
                    decision_function_shape='ovr', random_state=None)
Docstring:     
C-Support Vector Classification.

The implementation is based on libsvm. The fit time complexity
is more than quadratic with the number of samples which makes it hard
to scale to dataset with more than a couple of 10000 samples.

The multiclass support is handled according to a one-vs-one scheme.

SVR
    Support Vector Machine for Regression implemented using libsvm.

LinearSVC
    Scalable Linear Support Vector Machine for classification
    implemented using liblinear. Check the See also section of
    LinearSVC for more comparison element.
    
"""

clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) 

clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) 

"""
DummyClassifier?
Init signature: DummyClassifier(strategy='stratified', 
                                random_state=None, constant=None)
Docstring:
DummyClassifier is a classifier that makes predictions using simple rules.

This classifier is useful as a simple baseline to compare with other
(real) classifiers. Do not use it for real problems.

DummyClassifier implements several such simple strategies for classification:

    - stratified generates random predictions by respecting the 
        training set class distribution.
    - most_frequent always predicts the most frequent label
        in the training set.
    - prior always predicts the class that 
        maximizes the class prior (like most_frequent`) and 
        ``predict_proba returns the class prior.
    - uniform generates predictions uniformly at random.
    - constant always predicts a constant label that is provided by the user.
        A major motivation of this method is F1-scoring, 
        when the positive class is in the minority.

Note that with all these strategies, the 
    predict method completely ignores the input data!

DummyRegressor also implements four simple rules of thumb for regression:

mean always predicts the mean of the training targets.
median always predicts the median of the training targets.
quantile always predicts a user provided quantile of the training targets.
constant always predicts a constant value that is provided by the user.

In all these strategies, the predict method completely 
    ignores the input data.

"""

# compare the accuracy of SVC and most_frequent
clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)



