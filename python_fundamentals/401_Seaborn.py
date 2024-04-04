# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:16:32 2024

@author: jampa
"""

import seaborn as sns

## For Numerical Values

#distplot
#joinplot
#pairplot

#------------------------------
##distplot
#------------------------------

df = sns.load_dataset("tips")
df.head()

#Corrleation with Heat map -> only for values floating or integer types
df.dtypes

#A Corrleation heatmap used coloured cells, typically in a monochromatic scala to show a 2D 
# correlation matrix(table) between two disscrete dimensions or event types

df.corr()
sns.heatmap(df.corr())

#------------------------------
##jointplot
#------------------------------

# Univariate Analysis

# A joint plot allows to study the relationship between 2 numeric variables. The central chart display their corrleation.
# It is usually a scatterplot, a hexabin plot, a 2D histogram or a 2D density plot

sns.jointplot(data=df,x='tip',y='total_bill',kind="hex")


sns.jointplot(data=df,x='tip',y='total_bill',kind="reg")


#------------------------------
##pairplot
#------------------------------

#A "pairs plot" is also known as scatterplot, in which one variable in the same data is matched with 
#another variables value, like this: Pairs plot are just elaborations on this, showing all variables paired with all other

sns.pairplot(df)


sns.pairplot(df, hue="sex")


#------------------------------
## dist plot
#------------------------------
# helps you to check distribution of a column feature

sns.displot(df["tip"])

sns.displot(df["tip"],kde=True,bins=10)


#-----------------------------------------------------------------------------
## categorial plots
#-----------------------------------------------------------------------------

#seaborn also helps us in doing the analysis on Categorial data points.

#boxplot
#violinplot
#countplot
#bar plot

#-----------------------------------------------------------------------------
## count plots
#-----------------------------------------------------------------------------

# for one feature gives the info

sns.countplot(data=df,x='sex')


#-----------------------------------------------------------------------------
## bar plots
#-----------------------------------------------------------------------------

sns.barplot(data=df, y="total_bill", x="sex")

#-----------------------------------------------------------------------------
## box plots
#-----------------------------------------------------------------------------

#It is a graph that represents information from a five-number summary

sns.boxplot(data=df, x='smoker', y='total_bill')


sns.boxplot(data=df, x='sex', y='total_bill')


sns.boxplot(data=df, orient='v') # shows only integer or flat values

#categorize based on some other categories

sns.boxplot(data=df, x='day', y='total_bill', hue='smoker')


#-----------------------------------------------------------------------------
## voilin plots
#-----------------------------------------------------------------------------

#violin plots helps us to see both the distribution of data in terms of kernel density estimation and the box plot

sns.violinplot(data=df, x='day', y='total_bill')