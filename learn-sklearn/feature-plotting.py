
from pandas.plotting import parallel_coordinates
'''
import pandas.tools.plotting
dir(pandas.tools.plotting)
 'andrews_curves',
 'autocorrelation_plot',
 'bootstrap_plot',
 'boxplot',
 'deregister_matplotlib_converters',
 'lag_plot',
 'm',
 'outer',
 'parallel_coordinates',
 'plot_params',
 'radviz',
 'register_matplotlib_converters',
 'scatter_matrix',
 'sys',
 't',
 'table',
 'warnings']
'''



"""
parallel_coordinates?
parallel_coordinates(frame, class_column, cols=None, ax=None, color=None, 
                     use_columns=False, xticks=None, colormap=None, axvlines=True, 
                     axvlines_kwds=None, sort_labels=False, **kwds)
"""
from sklearn.datasets import (load_iris)
iris = load_iris()
target = pd.Series([iris.target_names[i] for i in iris.target],name='Name')
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X = pd.concat([X,target],axis=1)

import itertools

col_order=['petal length (cm)',
           'petal width (cm)',
           'sepal length (cm)',
           'sepal width (cm)']

for order in itertools.permutations(col_order):
    print(list(order))
    parallel_coordinates(X, class_column='Name', cols=list(order))
    plt.show()












