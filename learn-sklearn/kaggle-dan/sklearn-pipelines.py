
"""
pipeline bundles preprocessing and modeling steps 
so you can use the whole bundle as if it were a single step

Cleaner Code: You won't need to keep track of your 
    training (and validation) data at each step of processing. 
    Accounting for data at each step of processing can get messy. 
    With a pipeline, you don't need to manually keep track of each step.
Fewer Bugs: There are fewer opportunities to mis-apply a step or 
    forget a pre-processing step.
Easier to Productionize: It can be surprisingly hard to 
    transition a model from a prototype to something 
    deployable at scale. We won't go into the many related 
    concerns here, but pipelines can help.
More Options For Model Testing: You will see an 
    example in the next tutorial, 
    which covers cross-validation.

Transformers are for pre-processing before modeling.
Over time, you will learn many more transformers, 
and you will frequently use multiple transformers sequentially

Models are used to make predictions. You will usually 
preprocess your data (with transformers) before putting it in a model.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer


# Read Data
data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
train_X, test_X, train_y, test_y = train_test_split(X, y)

my_imputer = Imputer()
my_model = RandomForestRegressor()

imputed_train_X = my_imputer.fit_transform(train_X)
my_model.fit(imputed_train_X, train_y)
imputed_test_X = my_imputer.transform(test_X)
predictions = my_model.predict(imputed_test_X)

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
"""
my_pipeline.fit(X, y=None, **fit_params)

Pipeline(memory=None,
     steps=[('imputer', Imputer(axis=0, copy=True, 
                            missing_values='NaN', strategy='mean', 
                            verbose=0)), 
     ('randomforestregressor', RandomForestRegressor(bootstrap=True, 
                                 criterion='mse', max_depth=None,
                                 max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, 
                                 min_impurity_s...timators=10, n_jobs=1,
                                 oob_score=False, 
                                 random_state=None, verbose=0, 
                                 warm_start=False))])

RandomForestRegressor?
"""
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

# modify fit arguments
my_pipeline = make_pipeline(Imputer(), 
                            XGBRegressor(xgbregressor__early_stopping_rounds = 5))
my_pipeline.fit(train_X, train_y)





