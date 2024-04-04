# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:42:42 2022

@author: jampa

Chi Square Test
"""

#import packages
import pandas as pd
from scipy.stats import chi2_contingency, chi2

#import data
campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name = "campaign_data")

#filter our data
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]

print(campaign_data["mailer_type"].value_counts())

print(campaign_data["signup_flag"].value_counts())

print(campaign_data[(campaign_data["mailer_type"] == "Mailer1") & (campaign_data["signup_flag"] == 1)])

#summarize to get observed frequencies

observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values

mailer1_signup_rate = 123 / (252 + 123)
mailer2_signup_rate = 127 / (209 + 127)

# state hypothesis and set acceptance criteria
null_hypothesis = "There is no relationship between mailer type and signup rate. They are independent"
alternate_hypothesis = "There is a relationship between mailer type and signup rate. They are not independent"
acceptance_criteria = 0.05

# calculate expected frequencies & CHI square statistic

chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False) #if dof (degrees of freedim) is 1 then set correction to false
print(chi2_statistic, p_value)


# Find the critical value for our test

critical_value = chi2.ppf(1 - acceptance_criteria, dof)         #percetage point function
print(critical_value)

# Print the results

if chi2_statistic >= critical_value:
    print(f" As our chi-square statistic of {chi2_statistic} is higher than critical value of {critical_value} we reject the null hypothesis and conclude that: {alternate_hypothesis}")
else:
    print(f" As our chi-squar of {chi2_statistic} is lower than critical value of {critical_value} we retain the null hypothesis and conclude that: {null_hypothesis}")
    
# Print the results (p-value)


if p_value <= acceptance_criteria:
    print(f" As our p-value of {p_value} is lower than acceptance_criteria of {acceptance_criteria} we reject the null hypothesis and conclude that: {alternate_hypothesis}")
else:
    print(f" As our p-value of {p_value} is higher than acceptance_criteria of {acceptance_criteria} we retain the null hypothesis and conclude that: {null_hypothesis}")
    
