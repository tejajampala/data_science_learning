########################################################
# Random Forest for Classification Tree - Basic Template
########################################################

# Import required python packages

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Import sample data

my_df = pd.read_csv("data/sample_data_classification.csv")

# Split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify = y)

# Instantiate our model object

clf = RandomForestClassifier(random_state = 42)

# Train our Model

clf.fit(X_train, y_train)

# Assess model accuracy

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


                 





