
# =============================================================================
# https://www.kaggle.com/learn/machine-learning
# =============================================================================

melbourne_data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\melb_data.csv',sep=',')
melbourne_data.describe()
melbourne_data.shape
#melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_predictors]

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

dir(DecisionTreeRegressor)

# =============================================================================

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
y.head()

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
mean_absolute_error?

# =============================================================================

train_X, eval_X, train_y, eval_y = train_test_split(X, y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
eval_predictions = melbourne_model.predict(eval_X)
print(mean_absolute_error(eval_y, eval_predictions)) # 249676.09317623958

# =============================================================================
"""
DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')
"""
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes => varying tree depth
cross_valid_result = pd.DataFrame(columns=['max_leaf_nodes','MAE'])
for max_leaf_nodes in range(100,2000,100):
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    cross_valid_result = cross_valid_result.append(dict(max_leaf_nodes=max_leaf_nodes,
                                                        MAE=my_mae),ignore_index=True)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# max_leaf_nodes=500 is optimal
cross_valid_result.plot(x='max_leaf_nodes',y='MAE',kind='line')

# =============================================================================

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(eval_X)
print(mean_absolute_error(eval_y, melb_preds)) # 188447.5712125675



