
"""
If you are familiar with linear or logistic regression models, 
partial dependence plots can be interepreted similarly 
to the coefficients in those models. But partial dependence 
plots can capture more complex patterns from your data, 
and they can be used with any model.

Partial dependence plots show how each variable or 
predictor affects the model's predictions. This is useful for questions like:

Sanity check the model is giving realistic interpretation
    
How much of wage differences between men and women are 
due solely to gender, as opposed to differences in education 
backgrounds or work experience?

We first predict the price for that house when sitting 
distance to 4. We then predict it's price setting distance to 5. 
Then predict again for 6. And so on. We trace out how predicted 
price changes (on the vertical axis) as we move from small values 
of distance to large values (on the horizontal axis).

But because of interactions, the partial dependence plot for a 
single house may be atypical. So, instead we repeat that 
mental experiment with multiple houses, and we plot the 
average predicted price on the vertical axis. You'll see some 
negative numbers. That doesn't mean the price would sell for a 
negative price. Instead it means the prices would have 
been less than the actual average price for that distance. 

"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\melb_data.csv')
y = data.Price
cols_to_use = ['Distance', 'Landsize', 'BuildingArea']
X = data[cols_to_use]
my_imputer = Imputer()
imputed_X = my_imputer.fit_transform(X)

my_model = GradientBoostingRegressor()
my_model.fit(imputed_X, y)
plot_partial_dependence?
"""
plot_partial_dependence(gbrt, X, features, feature_names=None, 
                        label=None, n_cols=3, grid_resolution=100, 
                        percentiles=(0.05, 0.95), n_jobs=1, verbose=0, 
                        ax=None, line_kw=None, contour_kw=None, **fig_kw)
Parameters
----------
gbrt : BaseGradientBoosting
    A fitted gradient boosting model.
X : array-like, shape=(n_samples, n_features)
    The data on which ``gbrt`` was trained.
features : seq of ints, strings, or tuples of ints or strings
    If seq[i] is an int or a tuple with one int value, a one-way
    PDP is created; if seq[i] is a tuple of two ints, a two-way
    PDP is created.
    If feature_names is specified and seq[i] is an int, seq[i]
    must be < len(feature_names).
    If seq[i] is a string, feature_names must be specified, and
    seq[i] must be in feature_names.
feature_names : seq of str
    Name of each feature; feature_names[i] holds
    the name of the feature with index i.
label : object
    The class label for which the PDPs should be computed.
    Only if gbrt is a multi-class model. Must be in ``gbrt.classes_``.
n_cols : int
    The number of columns in the grid plot (default: 3).
percentiles : (low, high), default=(0.05, 0.95)
    The lower and upper percentile used to create the extreme values
    for the PDP axes.
grid_resolution : int, default=100
    The number of equally spaced points on the axes.
"""
my_plots = plot_partial_dependence(gbrt=my_model, 
                                   features=[0,1,2],
                                   X=imputed_X,
                                   feature_names=cols_to_use, 
                                   grid_resolution=100,
                                   n_cols=2)

partial_dependence?
"""
plot yourself with seaborn...

partial_dependence(gbrt, target_variables, grid=None, 
                   X=None, percentiles=(0.05, 0.95), 
                   grid_resolution=100)
"""
p_dep = partial_dependence(gbrt=my_model,
                           target_variables=[0,1,2],
                           X=imputed_X,
                           percentiles=(0.05,0.95))

p_dep[0][0] #y or PDP
p_dep[1][0] #x for Distance
p_dep[1][1] #x for Land
p_dep[1][2] #x for BuildingArea














