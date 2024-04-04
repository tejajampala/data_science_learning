# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:46:33 2024

@author: jampa
"""
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = load_breast_cancer()

X = pd.DataFrame(df['data'], columns=df['feature_names'])
y = pd.DataFrame(df["target"],columns=["target"])

X.head()


##############################
# Split out Training and Test sets
##############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)





