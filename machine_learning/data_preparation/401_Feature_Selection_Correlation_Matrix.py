# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:31:32 2022

@author: jampa
"""

import pandas as pd

my_df = pd.read_csv("C:/Home/Data_Science_Infinity/machine_learning/data_preparation/feature_selection_sample_data.csv")

correlation_matrix = my_df.corr()