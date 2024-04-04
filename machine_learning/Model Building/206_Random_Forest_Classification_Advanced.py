
###################################################
#Random Forest For Classification - ABC Grocery Task
###################################################

###################################################
#import required packages
###################################################
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

###################################################
#import data
###################################################
data_for_model = pickle.load(open("data/abc_classification_modeling.p", "rb"))

###################################################
#drop unnecessary columns
###################################################
data_for_model.drop("customer_id", axis=1 , inplace = True)

###################################################
#shuffle data
###################################################
data_for_model = shuffle(data_for_model, random_state=42)

# Class Balance

data_for_model["signup_flag"].value_counts()
data_for_model["signup_flag"].value_counts(normalize = True)

#################################################
#deal with missing values
#################################################
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace= True)

################################################
# Split Input Variables and Output Variables
################################################

X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]

##############################################
# Split out Training and Test sets
##############################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify = y)

############################################
# Deal with categorical values
############################################

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
clf = RandomForestClassifier(random_state = 42, n_estimators = 500, max_features= 5)
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


# Feature Importance - represents mean decrease in ginni score

feature_importance = pd.DataFrame(clf.feature_importances_)
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

result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)

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















