# -*- coding: utf-8 -*-
"""
Created on Sun May 29 00:44:15 2022

@author: jampa
"""

import pandas as pd

my_df = pd.DataFrame()
my_df = pd.DataFrame({"Name": ["Tom","Dick","Harry"]})

#using dictionary
my_df = pd.DataFrame({"Name": ["Tom","Dick","Harry"], "ID":[101,102,103]})
my_list = [["Tom", 101], ["Dick", 102], ["Harry", 103]]

#using lists
my_df = pd.DataFrame(my_list, columns=["Name", "ID"])
my_df = pd.read_csv("") #add the file name to read data

transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)

#---------------------------------
#Exploring and Understanding Data

transactions.head(20)
transactions.tail(20)
transactions.sample(10) #gets sample data
sample = transactions.sample(frac = 0.1) #gets sample fraction data, in this case 10%

transactions.describe() #how the data is spread

#shows the list of largest and smallest 25 values, helps to understand the data spread
transactions.nlargest(25,"sales_cost")
transactions.nsmallest(25,"sales_cost")

#unique values in the data
transactions.nunique()

transactions["customer_id"].value_counts()

#null or missing values
customer_details = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="customer_details")
print(customer_details)

#checks if value is null or present or not, outputs a boolean
customer_details.isna()

customer_details.isna().sum() #number of values with null or missing values

#isna & is null are the same

#------------------------------------------
# Accessing columns

transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)

new_df = transactions.customer_id #cannot select multiple columns with "." dot notation

new_df = transactions["customer_id"]
my_var = "customer_id"
transactions[my_var]

new_df = transactions[["customer_id"]]

new_df = transactions[["customer_id","sales_cost"]]

#------------------------------------------
# Adding & Dropping Columns

transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)

transactions["store_id"] = 1

transactions["profit"] = transactions["sales_cost"] * 0.2

import numpy as np

#where to apply conditions
transactions["sales_type"] = np.where(transactions["sales_cost"] > 20, "Large", "Small") 

#using select to add columns based on a condition
condition_rules = [transactions["sales_cost"] > 50, transactions["sales_cost"] > 20, transactions["sales_cost"] > 10]
outcomes = ["X-Large", "Large", "Medium"]
transactions["sales_type"] = np.select(condition_rules, outcomes, default = "Small")

#remove columns
new_df = transactions.drop(["customer_id","sales_cost"], axis = 1) #removes columns

#----------------
#Map

customer_details = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="customer_details")
print(customer_details)

product_areas = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="product_areas")
print(product_areas)


#all values need to mentioned in dictionary or else we get nan (null) values
customer_details["gender_numeric"] = customer_details["gender"].map({"M" : 0, "F":1})

#null is displayed for M Values as we did not map it
customer_details["gender_numeric"] = customer_details["gender"].map({"F":1}) 

#Replace
customer_details["gender_numeric"] = customer_details["gender"].replace({"M" : 0, "F":1})

#M is displayed as M, as the replace does not have value for it.
customer_details["gender_numeric"] = customer_details["gender"].replace({"F":1}) 

#Apply - Applies the function on a column by default, by adding axis we can change it to row 
# for a series

product_areas["product_area_name"].apply(len)

def update_profit_margin(profit_margin):
    if profit_margin > 0.2:
        return profit_margin * 1.2
    else:
        return profit_margin * 0.8
    
product_areas["profit_margin_updated"] = product_areas["profit_margin"].apply(update_profit_margin)

x = pd.DataFrame({"A":[1,2], "B":[3,4], "C":[5,6]})

x.apply(max)
x.apply(max, axis=1) #in row

#ApplyMap - Applies the function to each element of the data frame
# all data is same datatype

def square(n):
    return n ** 2

x.applymap(square)

#-------------------------
# sorting and ranking

customer_details = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="customer_details")
print(customer_details)

#sorting
customer_details.sort_values(by = "distance_from_store", inplace = True)

customer_details.sort_values(by = "distance_from_store", inplace = True, ascending=False)

#sort by multiple columns
customer_details.sort_values(by = ["distance_from_store","credit_score"], inplace = True)

#mssing values are listed first, by defualt it is last
customer_details.sort_values(by = "distance_from_store", inplace = True, na_position = "first")

#Ranking
import numpy as np
x = pd.DataFrame({"column1" : [1,1,1,2,3,4,5,np.nan,6,8]})

x["column1"].rank()
x["column1_rank"] = x["column1"].rank()

x["average_rank"] = x["column1"].rank(method = "average")
x["min_rank"] = x["column1"].rank(method = "min")
x["max_rank"] = x["column1"].rank(method = "max")
x["first_rank"] = x["column1"].rank(method = "first")
x["dense_rank"] = x["column1"].rank(method = "dense")

#missing values are assigned a rank based on the na_position
x["dense_rank_na_top"] = x["column1"].rank(method = "dense", na_position="top")
x["dense_rank_na_bottom"] = x["column1"].rank(method = "dense", na_position="bottom")

#------------------------
# Pandas LOC & ILOC

transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)

#transactions.loc[row_label, column_label]
#transactions.iloc[row_indexes, column_indexes]

#ILOC - using idexes
transactions.iloc[0]
transactions.iloc[0:4]
transactions.iloc[[0,31,50]] #list of row indexes
transactions.iloc[0:4,[0,3,-1]] #mix of rows and columsn

transactions.iloc[[0,5,7],[0,3,-1]] 
transactions.iloc[:,[0,3,-1]] #all rows
transactions.iloc[[0,3,-1],:] #all columns

#LOC - using lables

transactions.loc[0]
transactions.set_index("cutomer_id", inplace = True)
transactions.loc[642]

transactions.reset_index(inplace = True)
list(transactions) #gets the list of colum names

transactions.loc[0:10,"customer_id"]
transactions.loc[0:10,["customer_id", "product_area_id"]]

#conditional logic

transactions["customer_id"] == 642

transactions.loc[transactions["customer_id"] == 642, ["customer_id", "sales_cost"]]

#each condition should be in separate brackets
transactions.loc[(transactions["customer_id"] == 642) & (transactions["num_items"] > 5)]

transactions.loc[(transactions["customer_id"] == 642) | (transactions["num_items"] > 5)]

transactions.loc[transactions["customer_id"].isin([642,700])]

#negate
transactions.loc[~transactions["customer_id"].isin([642,700])]

#------------------------------
#Renaming columns
import pandas as pd

transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)

list(transactions)

transactions.rename(columns= {"customer_id":"friend_id"}, inplace = True)
list(transactions)

column_names = ['friend_id',
 'transaction_date',
 'transaction_id',
 'product_group_id',
 'num_items',
 'sales_cost']

transactions.columns = column_names
list(transactions)

column_names = ['friend id',
 'transaction date',
 'transaction id',
 'product_group id',
 'num items',
 'sales cost']

transactions.columns = column_names
list(transactions)

transactions.columns = transactions.columns.str.replace(" ","_")
list(transactions)

#-------------------------
#joining and merging dataframes

transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)

df_a = pd.DataFrame({"A":[1,2,3], "B":[4,5,6]})
df_b = pd.DataFrame({"C":[1,2,3], "D":[4,5,6]})

df_c = pd.concat([df_a,df_b], axis = 1)
df_c = pd.concat([df_a,df_b], axis = 0)

df_a = pd.DataFrame({"A":[1,2,3], "B":[4,5,6]})
df_b = pd.DataFrame({"A":[1,2,3], "B":[4,5,6]})

df_c = pd.concat([df_a,df_b], axis = 0)

df_a.append(df_b)

#merging

df_a = pd.DataFrame({"user_id":[1,2,3,5,7], "age":[10,40,6,10,12]})
df_b = pd.DataFrame({"user_id":[1,2,3,4,5], "gender":["m","f","f","f","m"]})

pd.merge(df_a, df_b, how="inner", on ="user_id")

pd.merge(df_a, df_b, how="left",on ="user_id")
         
pd.merge(df_a, df_b, how="outer", on="user_id")

pd.merge(df_a, df_b, how="outer", on=["user_id","age"]) #join on multiple columns

pd.merge(df_a, df_b, how="outer", left_on="user_id", right_on="user") #join on columns with different names

#----------------------
#Aggregating using group by

transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)

product_areas = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="product_areas")
print(product_areas)

transactions["sales_cost"].sum()

transactions = pd.merge(transactions, product_areas, how="inner", on ="product_area_id")

transactions["product_area_name"].value_counts()

transactions.groupby("product_area_name")["sales_cost"].sum()

transactions.groupby("product_area_name")["sales_cost"].quantile([0.25,0.5,0.75])

sales_summary = transactions.groupby("product_area_name")["sales_cost"].sum().reset_index()

sales_summary = transactions.groupby(["product_area_name","transaction_date"])["sales_cost"].sum().reset_index()

sales_summary = transactions.groupby(["product_area_name","transaction_date"])[["sales_cost","num_items"]].sum().reset_index()
    
#using agg
sales_summary = transactions.groupby("product_area_name")["sales_cost"].agg("sum").reset_index()

sales_summary = transactions.groupby("product_area_name")["sales_cost"].agg("sum","mean").reset_index()

sales_summary = transactions.groupby(["product_area_name","transaction_date"])[["sales_cost","num_items"]].agg("sum","mean").reset_index()

sales_summary = transactions.groupby("product_area_name").agg({"sales_cost":"sum","num_items":"mean"}).reset_index()

#---------------
#pivots


transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
print(transactions)
list(transactions)

product_areas = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="product_areas")
print(product_areas)

transactions = pd.merge(transactions, product_areas, how="inner", on ="product_area_id")

sales_summary = transactions.groupby(["product_area_name","transaction_date"])["sales_cost"].sum().reset_index()

sales_summary_pivot = transactions.pivot_table(index = "transaction_date",
                                               columns = "product_area_name",
                                               values = "sales_cost",
                                               aggfunc = "sum"
                                               )

print(sales_summary_pivot)
sales_summary_pivot.plot()


sales_summary_pivot = transactions.pivot_table(index = ["transaction_date", "profit_margin"],
                                               columns = "product_area_name",
                                               values = "sales_cost",
                                               aggfunc = "sum",
                                               fill_value = 0,
                                               margins = True,
                                               margins_name = "Total"
                                               )


#-------------------------------
#missing values


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

import numpy as np
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

                  
#------------------
#duplicates


my_df = pd.DataFrame({"customer_id": [1,1,2,2,3],
                      "transaction_id":[101,102,103,103,104]})

#gets the boolean rows with duplicated
my_df.duplicated()

#gets the count of rows that are duplicated
my_df.duplicated().sum()

my_df["customer_id"].duplicated()

#selects duplicate rows
my_df[my_df.duplicated()]

#consider first row as actual row
my_df.duplicated(keep = "first")

#consider the last row as the actual row
my_df.duplicated(keep = "last")

#removes rows that are duplicate
my_df.duplicated(keep = False)

#drop the duplicates
my_df.drop_duplicates(inplace = True)

#----------------------------
# Plotting our data using Pandas


transactions = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="transactions")
product_areas = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="product_areas")
customer_details = pd.read_excel("C:/Home/Data_Science_Infinity/python_fundamentals/grocery_database.xlsx", sheet_name="customer_details")

customer_details.plot() #plots a line  graph for all the columns, default is line

daily_sales_summary = transactions.groupby("transaction_date")[["sales_cost","num_items"]].sum().reset_index()

daily_sales_summary["sales_cost"].plot()

daily_sales_summary.plot(x = "transaction_date", y = "sales_cost")
daily_sales_summary.plot(x = "transaction_date", y = "sales_cost", kind="line")

daily_sales_summary.plot(x = "transaction_date", y = "sales_cost", kind="scatter")

#box plot on a single columns helps us to understand the data
daily_sales_summary.plot(y = "sales_cost", kind = "box")

daily_sales_summary.plot(y = "sales_cost", kind = "hist", bins=25)

product_areas.plot(kind = "bar", y = "profit_margin", x = "product_area_name")

#--------------------------------------
# exporting data


my_df = pd.DataFrame({"A": [1,2,3],
                      "B": ["one", np.nan, "three"]})

my_df.to_csv("tester_export.csv", index = False)

my_df.to_csv("tester_export.csv", index = False, columns=["B"])

my_df.to_csv("tester_export.csv", index = False, header = False)

my_df.to_csv("tester_export.csv", index = False, na_rep = "MISSING")

my_df.to_csv("tester_export.csv", index = False, na_rep = "MISSING")

my_df.to_csv("tester_export.csv", index = False, sep = "\t")

my_df.to_excel("tester_export.xlsx", sheet_name="Sheet_12345")

my_other_df = my_df * 3

with pd.ExcelWriter("tester_export.xlsx") as excel_writer:
    my_df.to_excel(excel_writer, sheet_name="Sheet_12345")
    my_other_df.to_excel(excel_writer, sheet_name="Sheet_6789")
    

#add r to the path while saving file in windows, as it helps with unicode issue
my_df.to_csv(r"C:\users\tester_export.csv", index = False)






from sklearn.model_selection import GridSearchCV

gscv = GridSearchCV(
       estimator = RandomForestClassifier(random_state=42),
       param_grid = {"n_estimators": [100, 500, 750, 1000, 1250, 1500],
                     "max_depth": [1,2,3,4,5,6,7,8,9,10, None]
                    },
       cv = 5,
       scoring = "r2",
       n_jobs = -1
       )
       











