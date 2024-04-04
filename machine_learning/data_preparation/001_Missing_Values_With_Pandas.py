# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:17:44 2022

@author: jampa
"""
import numpy as np
import pandas as pd

customer_details = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="customer_details")
print(customer_details)

customer_details.isna().sum() #counting the boolean values to check for missing values

customer_details.notna().sum()

customer_details["distance_from_store"].isna().sum()

customer_details[customer_details["distance_from_store"].isna()] #selects the values that are missing

customer_details.dropna(how = "any") #not selects any row that has na
customer_details.dropna(how = "all") #not selects rows only if all the columns are na

#to check for na only in subset of columns
customer_details.dropna(how = "any", subset =["distance_from_store"])
customer_details.dropna(how = "any", subset = ["distance_from_store", "gender"])


my_df = pd.DataFrame({"A": [1,2,4,np.nan,5,np.nan,7],
                      "B": [4,np.nan,7,np.nan,1,np.nan,2]})


my_df["A"].fillna(value = 0)
impute_value = my_df["A"].mean()

my_df["A"].fillna(value = impute_value)


customer_details.isna().sum()

customer_details["gender"].fillna(value = "U", inplace = True)
customer_details["gender"].value_counts()


customer_details["distance_from_store"].describe()

customer_details["distance_from_store"].info()

customer_details["distance_from_store"].fillna(value = customer_details["distance_from_store"].median, inplace = True)

                 
