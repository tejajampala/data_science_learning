# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:22:19 2022

@author: jampa
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

my_df = pd.DataFrame({"A":[1,4,7,10,13],
                      "B":[3,6,9,10,15],
                      "C":[2,5,7,11,np.nan]})
                
knn_imputer = KNNImputer()
knn_imputer = KNNImputer(n_neighbors = 1)
knn_imputer = KNNImputer(n_neighbors = 2)
knn_imputer = KNNImputer(n_neighbors = 2, weights = "distance")
knn_imputer.fit_transform(my_df)

my_df1 = pd.DataFrame(knn_imputer.fit_transform(my_df), columns = my_df.columns)
