
"""
leading model for working with standard tabular data

To reach peak accuracy, XGBoost models require more knowledge 
and model tuning than techniques like Random Forest. 

After this tutorial, you'ill be able to
    Follow the full modeling workflow with XGBoost
    Fine-tune XGBoost models for optimal performance

XGBoost is an implementation of the Gradient Boosted Decision Trees algorithm 
(scikit-learn has another version of this algorithm, 
but XGBoost has some technical advantages.)

We need some base prediction to start the cycle. 
In practice, the initial predictions can be pretty naive. 
Even if it's predictions are wildly inaccurate, 
subsequent additions to the ensemble will address those errors.

Cycle:
build model -> predicts -> reconstructed error -> build model on error
    -> add model to ensemble -> repeat to extend ensemble


"""

# install binary xgboost 
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost

from xgboost import XGBRegressor, XGBClassifier, XGBModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), 
                                                    y.as_matrix(), test_size=0.25)

Imputer?
#Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

XGBRegressor?
#XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, 
#             silent=True, objective='reg:linear', booster='gbtree', 
#             n_jobs=1, nthread=None, gamma=0, min_child_weight=1, 
#             max_delta_step=0, subsample=1, colsample_bytree=1, 
#             colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
#             scale_pos_weight=1, base_score=0.5, random_state=0, 
#             seed=None, missing=None, **kwargs)
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)
# make predictions
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# =============================================================================
# model tuning
# =============================================================================

"""
subtle but important trick for better XGBoost models:

Instead of getting predictions by simply adding up the predictions 
from each component model, we will multiply the predictions from 
each model by a small number before adding them in. 
This means each tree we add to the ensemble helps us less. 
In practice, this reduces the model's propensity to overfit.

So, you can use a higher value of n_estimators without overfitting

Use early stopping to find a good value for n_estimators.

a small learning rate (and large number of estimators) will 
yield more accurate XGBoost models, though it will also take 
the model longer to train since it does more iterations through the cycle

early stopping:
 smart to set a high value for n_estimators and 
 then use early_stopping_rounds to find the optimal time to stop iterating
 stop after K rounds of deteoriation of CV scores. 
 Note: 1 round of deteoriation can happen by chance. 
 require: training data + test data for CV

 n_estimators:
 how many cycle of model on error of error of error... 
 too low -> underfit       too high -> overfit

"""

for n_est in range(100,1100,100):
    my_model = XGBRegressor(n_estimators=n_est, learning_rate=0.05)
    my_model.fit(train_X, train_y, early_stopping_rounds=5, 
                 eval_set=[(test_X, test_y)], verbose=False)
    
    predictions = my_model.predict(test_X)
    print("Mean Absolute Error with {0}: ".format(n_est) + str(mean_absolute_error(predictions, test_y)))


for n_est in range(100,1100,100):
    my_model = XGBRegressor(n_estimators=n_est, learning_rate=0.05)
    my_model.fit(train_X, train_y, early_stopping_rounds=300, 
                 eval_set=[(test_X, test_y)], verbose=False)
    
    predictions = my_model.predict(test_X)
    print("Mean Absolute Error with {0}: ".format(n_est) + str(mean_absolute_error(predictions, test_y)))

for n_est in range(100,1100,100):
    my_model = XGBRegressor(n_estimators=n_est, learning_rate=0.05)
    my_model.fit(train_X, train_y, 
                 eval_set=[(test_X, test_y)], verbose=False)
    
    predictions = my_model.predict(test_X)
    print("Mean Absolute Error with {0}: ".format(n_est) + str(mean_absolute_error(predictions, test_y)))


