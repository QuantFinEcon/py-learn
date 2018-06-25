
"""
leakage causes a model to look accurate until you start 
making decisions with the model, and then the model becomes very inaccurate

two main types of leakage: Leaky Predictors and a Leaky Validation Strategies.

"""

"""
leaky predictors

took_antibiotic_medicine is frequently changed after the 
value for got_pneumonia is determined. This is target leakage

To prevent this type of data leakage, any variable 
updated (or created) after the target value is realized should be excluded.

aren't careful distinguishing training data from validation data
this happens if you run preprocessing (like fitting the Imputer 
for missing values) before calling train_test_split

There is no single solution that universally prevents leaky predictors. 
It requires knowledge about your data, case-specific inspection and common sense.

To screen for possible leaky predictors, look for columns 
that are statistically correlated to your target.

use pipelines and do your preprocessing inside the pipeline.
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-sklearn\\AER_credit_card_data.csv', 
                   true_values = ['yes'],
                   false_values = ['no'])
print(data.head())

data.shape # small dataset 1309 rows

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

y = data.card
X = data.drop(['card'], axis=1)
X.dtypes

# Since there was no preprocessing, we didn't need a pipeline here. Used anyway as best practice
modeling_pipeline = make_pipeline(RandomForestClassifier())
cv_scores = cross_val_score(modeling_pipeline, X, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean()) # too good to be true

expenditures_cardholders = data.expenditure[data.card]
expenditures_noncardholders = data.expenditure[~data.card]
print('Fraction of those who received a card with no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
print('Fraction of those who received a card with no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))

potential_leaks = ['expenditure', 'share']
X2 = X.drop(potential_leaks, axis=1)
cv_scores = cross_val_score(modeling_pipeline, X2, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean())


# correlation matrix
corr = data.corr()
corr_html = corr.style\
    .background_gradient()\
    .set_precision(3)\
    .render()
with open("./correlation_table.html","w") as f:
    f.write( corr_html )

fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)


