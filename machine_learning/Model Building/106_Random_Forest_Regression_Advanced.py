######################################################
# Random Forest for Regression - Advanced Grocery Task
######################################################
# We do not need to remove outliers, as Random Forest do not care.
# Doing feature selection will no effect the acuracy of Random Forest, but will help in computation.
# Random Foreest are prone to overfitting, as they learn the training data very well

##############################
#import required packages
##############################
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
#from sklearn.feature_selection import RFECV

##############################
#import data
##############################
data_for_model = pickle.load(open("data/abc_regression_modeling.p", "rb"))

##############################
#drop unnecessary columns
##############################
data_for_model.drop("customer_id", axis=1 , inplace = True)

##############################
#shuffle data
##############################
data_for_model = shuffle(data_for_model, random_state=42)

##############################
#deal with missing values
##############################
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace= True)

##############################
# Split Input Variables and Output Variables
##############################

X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

##############################
# Split out Training and Test sets
##############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

##############################
# Deal with categorical values
##############################

categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

######################
# Model Training
######################
regressor = RandomForestRegressor(random_state = 42)
regressor.fit(X_train,  y_train)

#predict on the test set
y_pred = regressor.predict(X_test)

#calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

#cross validation
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv,  scoring = "r2")
cv_scores.mean()

#calculated Adjusted R-squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) /(num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Feature Importance

feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)


plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()


# Permutation Importance - This is preferred
# calculated using out of bag roles -> these are run through the decision tree and calculate accuracy
# next randomize the column values -> run through the decision tree and calculate accuracy

result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)

permutation_importance = pd.DataFrame(result['importances_mean'])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"], permutation_importance_summary["permutation_importance"])
plt.title("Permuation importance of Random Forest")
plt.xlabel("Permuation Importance")
plt.tight_layout()
plt.show()


# Predictions under the hood

y_pred[0]
new_data = [X_test.iloc[0]]
regressor.estimators_

predictions = []
tree_count = 0

for tree in regressor.estimators_:
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1
    
print(predictions)
sum(predictions) / tree_count

import pickle

pickle.dump(regressor, open("data/random_forest_regression_model.p", "wb"))
pickle.dump(one_hot_encoder, open("data/random_forest_regression_ohe.p", "wb"))

