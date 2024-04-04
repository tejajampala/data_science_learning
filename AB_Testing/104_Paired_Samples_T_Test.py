# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:03:08 2022

@author: jampa

Paired Samples T-Test
"""

# Import required packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, norm 

# Create mock data
#loc is the mean, scale is the standard deviation
before = norm.rvs(loc = 500, scale = 100, size = 100, random_state = 42).astype(int)

np.random.seed(42)
after = before + np.random.randint(low = -50, high = 75, size = 100)

plt.hist(before, density = True, alpha = 0.5, label = "Before")
plt.hist(after, density = True, alpha = 0.5, label = "After")
plt.legend()
plt.show()

before_mean = before.mean()
after_mean = after.mean()
print(before_mean,after_mean)

# set the hypothesis and acceptance criteria

null_hypothesis = "The mean of the before sample is equal to the mean of the after sample"
alternate_hypothesis = "The mean of the before sample is different to the mean of the after sample"
acceptance_criteria = 0.05

# execute the hypothesis test

t_statistic, p_value = ttest_rel(before, after)
print(t_statistic, p_value)

# Print the results (p-value)

if p_value <= acceptance_criteria:
    print(f" As our p-value of {p_value} is lower than acceptance_criteria of {acceptance_criteria} we reject the null hypothesis and conclude that: {alternate_hypothesis}")
else:
    print(f" As our p-value of {p_value} is higher than acceptance_criteria of {acceptance_criteria} we retain the null hypothesis and conclude that: {null_hypothesis}")
    