xx# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:41:24 2022

@author: jampa
"""

import pandas as pd

my_df = pd.read_csv("feature_selection_sample_data.csv")

# Regression Template - output numeric

# select k-best & f_regression - a linear model that tests the indiviual effect of each of our input variables using statistics

from sklearn.feature_selection import SelectKBest, f_regression

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

feature_selector = SelectKBest(f_regression, k="all") #k- number of values to select
fit = feature_selector.fit(X,y) #calucaltes relationship scores

fit.pvalues_ # lower value is better
fit.scores_ # f-scores (higher value is better)


p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_names = pd.DataFrame(X.columns)
summary_stats = pd.concat([input_variable_names,p_values,scores], axis = 1)
summary_stats.columns = ["input_variable","p_value", "f_score"]
summary_stats.sort_values(by = "p_value", inplace = True)


p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats["f_score"] >= score_threshold) & (summary_stats["p_value"] <= p_value_threshold)]
selected_variables = selected_variables["input_variable"].tolist()
X_new = X[selected_variables]

"""this can be used when we know the number of inputs to use
feature_selector = SelectKBest(f_regression, k=2)
fit = feature_selector.fit(X,y)
X_new1 = feature_selector.transform(X)
feature_selector.get_support()
X_new1 = X.loc[:,feature_selector.get_support()] 
"""

# Classification Template 

from sklearn.feature_selection import SelectKBest, chi2

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

feature_selector = SelectKBest(chi2, k="all") #k- number of values to select
fit = feature_selector.fit(X,y) #calucaltes relationship scores

fit.pvalues_ # lower value is better
fit.scores_ # f-scores (higher value is better)


p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_names = pd.DataFrame(X.columns)
summary_stats = pd.concat([input_variable_names,p_values,scores], axis = 1)
summary_stats.columns = ["input_variable","p_value", "chi2_score"]
summary_stats.sort_values(by = "p_value", inplace = True)


p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats["chi2_score"] >= score_threshold) & (summary_stats["p_value"] <= p_value_threshold)]
selected_variables = selected_variables["input_variable"].tolist()
X_new = X[selected_variables]
