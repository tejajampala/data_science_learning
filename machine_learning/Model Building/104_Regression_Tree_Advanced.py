
##############################
# Regression Tree - Advanced Grocery Task
##############################
# we do not need to remove outliers, as decision tress do not care.
# doing feature selection will no effect the acuracy of Decision Tree, but will help in computation.
# removing in the file as we have very less variables/features
# Decision Trees are prone to overfitting, as they learn the training data very well

##############################
#import required packages
##############################
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
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
regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
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

# A demonstration of overfitting - 
y_pred_training = regressor.predict(X_train)
print(r2_score(y_train, y_pred_training))

#plot our decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Finding the best max_depth
max_depth_list = list(range(1,9))
accuracy_scores = []

for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train,  y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

max_accuracy = max(accuracy_scores)
max_accuracy_idx =  accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]


# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = 'x', color = 'red')
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

#plot our model
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 16)







