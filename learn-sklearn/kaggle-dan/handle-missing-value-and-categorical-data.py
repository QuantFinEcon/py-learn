
# Load data
melb_data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer

melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)
list(melb_predictors.columns)
melb_predictors.describe()

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
melb_categorical_predictors = melb_predictors.select_dtypes(include=['object'])

# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

# =============================================================================
# drop columns
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

# =============================================================================
# Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
Imputer?
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# =============================================================================
"""
determined by whether rows with missing values are intrinsically
like or unlike those without missing values
"""
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

# =============================================================================
# encoding categorical var before input to model
# =============================================================================

train_data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\train.csv')
test_data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\test.csv')
test_data.dtypes
train_data.dtypes

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which drops these columns. 
cols_with_missing = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + 
                                             cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. 
# This is convenient, though a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# one-hot encoding: only 0 or 1, but longitudinal to cross tab with more columns
pd.get_dummies?
"""
Signature: pd.get_dummies(data, prefix=None, prefix_sep='_', 
                          dummy_na=False, columns=None, 
                          sparse=False, drop_first=False)
Docstring:
Convert categorical variable into dummy/indicator variables
"""
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
train_predictors.dtypes
one_hot_encoded_training_predictors.dtypes
train_predictors['KitchenQual'].unique()
one_hot_encoded_training_predictors.select_dtypes(include=['uint8']).describe()
train_predictors.select_dtypes(include=['object']).describe()

cross_val_score?
def get_mae(X, y):
    # multiple by -1 to make positive MAE score 
    # instead of neg value returned as sklearn convention
    return -1 * cross_val_score(estimator=RandomForestRegressor(n_estimators=50),
                                X=X, y=y, 
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

# train and test cardinality for columns might differ
# one-hot encoding leads to missing columns
# align columns presence and order. 
# Left align => all columns in training preserved
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

set(one_hot_encoded_training_predictors.columns)-set(one_hot_encoded_test_predictors.columns)
set(one_hot_encoded_test_predictors.columns)-set(one_hot_encoded_training_predictors.columns)
set(final_train)-set(final_test)
set(final_test)-set(final_train)

# =============================================================================
# OneHotEncoder
# =============================================================================
# cannot work on object,text data
# only handles nominal/categorical features encoded as columns of integers.
# can use with sklearn.pipeline.Pipeline
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
#OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
#       handle_unknown='error', n_values='auto', sparse=True)
enc.n_values_
enc.feature_indices_
enc.active_features_ # == sum(enc.n_values_)
enc.transform?
enc.transform([[0, 1, 1]]).toarray()

# =============================================================================
# Categoricals with Many Values
# =============================================================================
# uses the hashing trick to store high-cardinality columns / rows
from sklearn.feature_extraction import FeatureHasher

#turns sequences of symbolic feature names (strings) into scipy.sparse matrices
#hash function employed is the signed 32-bit version of Murmurhash3.
#low-memory alternative to DictVectorizer and CountVectorizer, 
#intended for large-scale (online) learning and situations 
#where memory is tight, e.g. when running prediction code on embedded devices.

h = FeatureHasher(n_features=10)
#    class sklearn.feature_extraction.FeatureHasher(n_features=1048576, 
#    input_type=’dict’, dtype=<class ‘numpy.float64’>, 
#    alternate_sign=True, non_negative=False)
D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
f = h.transform(D)
f.toarray()


