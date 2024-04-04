# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:43:46 2024

@author: jampa
"""

#from sklearn.datasets import load_boston

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", , header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#convert to dataframe
data_df = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]) 
target_df = pd.DataFrame(target, columns = ["PRICE"])

#build complete dataframe
full_df = pd.concat([data_df, target_df], axis = 1)


#Import sklearn modules
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV


#shuffle data
full_df = shuffle(full_df, random_state=42)


##############################
# Split Input Variables and Output Variables
##############################

X = full_df.drop(["PRICE"], axis = 1)
y = full_df["PRICE"]

##############################
# Split out Training and Test sets
##############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


##############################################################
# Linear Regression
##############################################################


######################
# Feature Selection
#######################

regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X_train, y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:,feature_selector.get_support()]
X_test = X_test.loc[:,feature_selector.get_support()]

plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

######################
# Model Training
######################
regressor = LinearRegression()
regressor.fit(X_train,  y_train)

#predict on the test set
y_pred = regressor.predict(X_test)

#calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

#cross validation
cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv,  scoring = "r2")
cv_scores.mean()

#calculated Adjusted R-squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) /(num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

#Extract Model Coefficients

coeficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names,coeficients], axis = 1)
summary_stats.columns = ["input_variable", "coefficient"]

#Extract Model Intercept

regressor.intercept_


##############################################################
# Ridge Regression
##############################################################
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
params = {"alpha" : [1e-15, 1e-10, 1e-08, 1e-3, 1e-2, 1, 5, 10, 20, 25, 50, 100]}
                     
ridge_regressor=GridSearchCV(ridge, params, scoring="r2", cv = 5)
ridge_regressor.fit(X_train, y_train)

ridge_regressor.best_params_
ridge_regressor.best_score_

#predict on the test set
regressor = ridge_regressor.best_estimator_
y_pred = regressor.predict(X_test)

#calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

#adjust r-squared
num_data_points, num_input_vars = X_test.shape
r_squared = lasso_regressor.best_score_
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) /(num_data_points - num_input_vars - 1)
print(adjusted_r_squared)


##############################################################
# Lasso Regression
##############################################################
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso = Lasso()
params = {"alpha" : [1e-15, 1e-10, 1e-08, 1e-3, 1e-2, 1, 5, 10, 20, 25, 50, 100]}
                     
lasso_regressor=GridSearchCV(lasso, params, scoring="r2", cv = 5)
lasso_regressor.fit(X_train, y_train)

lasso_regressor.best_params_
lasso_regressor.best_score_

#predict on the test set
regressor = lasso_regressor.best_estimator_
y_pred = regressor.predict(X_test)

#calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)

#adjust r-squared
num_data_points, num_input_vars = X_test.shape
r_squared = lasso_regressor.best_score_
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) /(num_data_points - num_input_vars - 1)
print(adjusted_r_squared)


