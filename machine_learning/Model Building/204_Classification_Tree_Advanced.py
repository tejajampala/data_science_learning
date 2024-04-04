
#########################################
#Classification Tree - Advanced Grocery Task
#########################################

##############################
#import required packages
##############################
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
#from sklearn.feature_selection import RFECV - not required

##############################
#import data
##############################
data_for_model = pickle.load(open("data/abc_classification_modeling.p", "rb"))

##############################
#drop unnecessary columns
##############################
data_for_model.drop("customer_id", axis=1 , inplace = True)

##############################
#shuffle data
##############################
data_for_model = shuffle(data_for_model, random_state=42)

# Class Balance

data_for_model["signup_flag"].value_counts()
data_for_model["signup_flag"].value_counts(normalize = True)

##############################
#deal with missing values
##############################
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace= True)

##############################
# Split Input Variables and Output Variables
##############################

X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]

##############################
# Split out Training and Test sets
##############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify = y)

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
clf = DecisionTreeClassifier(random_state = 42, max_depth = 5)
clf.fit(X_train,  y_train)

######################
# Model Assessment
######################

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test) [:,1]

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)
#print(conf_matrix)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i , j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()    


# Accuarcy (the number of correct classication out of all attempted classifications)
accuracy_score(y_test, y_pred_class)

# Precision (of all the observations that were predicted as positive, how many are actually positive)
#(tp/tp+fp)
precision_score(y_test, y_pred_class)

# Recall (of all observations, how many we predict as positive)
#(tp/tp+fn)
recall_score(y_test, y_pred_class)

# F1-Score (The harmonic mean of precision and recall)
# 2 * (precision * recall / (precision + recall))
f1_score(y_test, y_pred_class)


# Finding the best max_depth
max_depth_list = list(range(1,15))
accuracy_scores = []

for depth in max_depth_list:
    
    clf = DecisionTreeClassifier(max_depth = depth, random_state = 42)
    clf.fit(X_train,  y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred) ###################### f1-score
    accuracy_scores.append(accuracy)

max_accuracy = max(accuracy_scores)
max_accuracy_idx =  accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]


# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = 'x', color = 'red')
plt.title(f"Accuracy (F1 Score) by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy (F1 Score)")
plt.tight_layout()
plt.show()

#plot our model
plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 16)















