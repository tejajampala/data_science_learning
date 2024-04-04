# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 23:43:17 2022

@author: jampa
"""

import matplotlib.pyplot as plt

x_values = [0,1,2,3,4,5,6,7,8,9,10]
x_squared = [x ** 2 for x in x_values]

plt.plot(x_values, x_squared)
plt.title("Exponential Growth")
plt.xlabel("The values of x")
plt.ylabel("The values of y")
plt.show()

import pandas as pd

my_df = pd.DataFrame({"X": x_values
                      ,"Y": x_squared})

plt.plot(my_df["X"], my_df["Y"])
plt.title("Exponential Growth")
plt.xlabel("The values of x")
plt.ylabel("The values of y")
plt.show()

#--------------------------------------
# formatting

x_values = [0,1,2,3,4,5,6,7,8,9,10]
x_squared = [x ** 2 for x in x_values]
x_cubed = [x ** 3 for x in x_values]

plt.plot(x_values, x_squared, label = "X Squared")
plt.plot(x_values, x_cubed, label = "X cubed")
plt.title("Exponential Growth")
plt.xlabel("The values of x")
#plt.xticks([])
plt.ylabel("The values of y")
#plt.yticks([])
plt.grid(True)
plt.legend()
plt.show()

#--------------------------------
# Formatting our plots: Colours and Styles


x_values = [0,1,2,3,4,5,6,7,8,9,10]
x_squared = [x ** 2 for x in x_values]
x_cubed = [x ** 3 for x in x_values]

#matplotlib named colours - search for this to get diff colors. 
#Colours can be give a name or a list of RGB color = [1.0, 2, 0.25], values can be from 0 to 1
plt.plot(x_values, x_squared, label = "X Squared", color = "deeppink", linewidth = 3, linestyle = "--")
plt.plot(x_values, x_cubed, label = "X cubed", color = "#0000FF", linewidth = 3, linestyle = "--"  )
plt.title("Exponential Growth")
plt.xlabel("The values of x")
#plt.xticks([])
plt.ylabel("The values of y")
#plt.yticks([])
plt.grid(True)
plt.legend()
plt.show()


#markers - many markers check the documentation
#markeredgecolor & markerfacecolor
plt.plot(x_values, x_squared, label = "X Squared", color = "deeppink", linewidth = 2, marker = ".")
plt.plot(x_values, x_cubed, label = "X cubed", color = "#0000FF", linewidth = 2, marker = "o" , markersize = 10, markerfacecolor = "red", markeredgecolor = "green")
plt.title("Exponential Growth")
plt.xlabel("The values of x")
#plt.xticks([])
plt.ylabel("The values of y")
#plt.yticks([])
plt.grid(True)
plt.legend()
plt.show()

#different styles available
plt.style.available
plt.style.use('seaborn-poster')

plt.plot(x_values, x_squared, label = "X Squared")
plt.plot(x_values, x_cubed, label = "X cubed")
plt.title("Exponential Growth")
plt.xlabel("The values of x")
#plt.xticks([])
plt.ylabel("The values of y")
#plt.yticks([])
plt.grid(True)
plt.legend()
plt.show()

#-------------------------------------------
# working with subplots
# Matplotlib subplots

#plt.subplot(row,column,plot_number)
plt.subplot(2,1,1)
plt.plot(x_values, x_squared, label = "X Squared")
plt.title("Squared Values")
plt.xlabel("x")
plt.ylabel("y")


plt.subplot(2,1,2)
plt.plot(x_values, x_cubed, label = "X cubed")
#bar
#plt.bar(x_values, x_cubed, label = "X cubed")
plt.title("Cubed Values")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

#--------------------------------------------
# creating and refining a histogram

import matplotlib.pyplot as plt
import pandas as pd

body_data = pd.read_csv("C:\Home\Data_Science_Infinity\python_fundamentals\weights_and_heights.csv")
body_data.describe()

male = body_data[body_data["Gender"] == "Male"]
female = body_data[body_data["Gender"] == "Female"]

plt.style.use("seaborn-poster")
plt.hist(male["Weight"], bins =20, edgecolor = "black", alpha = 0.6, color = "royalblue", label ="Male")
plt.hist(female["Weight"], bins =20, edgecolor = "black", alpha = 0.6, color = "magenta", label ="Female")
plt.xlabel("Weight (lbs)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

#--------------------------------------------
# creating and refining a scatter plot

import matplotlib.pyplot as plt
import pandas as pd

body_data = pd.read_csv("C:\Home\Data_Science_Infinity\python_fundamentals\weights_and_heights.csv")
body_data.describe()

male = body_data[body_data["Gender"] == "Male"]
female = body_data[body_data["Gender"] == "Female"]

male_sample = male.sample(200, random_state = 42)

plt.style.use("seaborn-poster")
plt.scatter(male_sample["Weight"], male_sample["Height"], color = "blue", s = 700, alpha = 0.5)
plt.title("Weight vs Height for Males")
plt.xlabel("Weight (lbs)")
plt.ylabel("Height (in)")
plt.axvline(x = male_sample["Weight"].median(), color = "black", linestyle = "--")
plt.axhline(y = male_sample["Height"].median(), color = "black", linestyle = "--")
plt.tight_layout()
plt.show()

#-------------------------------
# enhancing plots using visual aids


import matplotlib.pyplot as plt
import pandas as pd

body_data = pd.read_csv("C:\Home\Data_Science_Infinity\python_fundamentals\weights_and_heights.csv")
body_data.describe()

male = body_data[body_data["Gender"] == "Male"]
female = body_data[body_data["Gender"] == "Female"]

male_sample = male.sample(200, random_state = 42)
patient = male.loc[[705]]

# s is the size, alpha is the transparency
plt.style.use("seaborn-poster")
plt.scatter(male_sample["Weight"], male_sample["Height"], color = "blue", s = 700, alpha = 0.5)
plt.scatter(patient["Weight"], patient["Height"], color = "pink", s = 700, alpha = 1, edgecolor = "red", linewidth=2)
plt.scatter(patient["Weight"], patient["Height"], marker = "x", color = "red", s = 250, alpha = 1, linewidth=5)
plt.title("Weight vs Height for Males")
plt.xlabel("Weight (lbs)")
plt.ylabel("Height (in)")
plt.axvline(x = male_sample["Weight"].median(), color = "black", linestyle = "--")
plt.axhline(y = male_sample["Height"].median(), color = "black", linestyle = "--")
plt.tight_layout()
plt.show()


#--------------------------------
# adding text to our plots

import matplotlib

print(matplotlib.__version__)


import matplotlib.pyplot as plt
import pandas as pd

body_data = pd.read_csv("C:\Home\Data_Science_Infinity\python_fundamentals\weights_and_heights.csv")
body_data.describe()

male = body_data[body_data["Gender"] == "Male"]
female = body_data[body_data["Gender"] == "Female"]

male_sample = male.sample(200, random_state = 42)
patient = male.loc[[705]]


median_weight = male_sample["Weight"].median()
median_height = male_sample["Height"].median()
min_weight = male_sample["Weight"].min()
min_height = male_sample["Height"].min()


# s is the size, alpha is the transparency
plt.style.use("seaborn-poster")
plt.scatter(male_sample["Weight"], male_sample["Height"], color = "blue", s = 700, alpha = 0.5)
plt.scatter(patient["Weight"], patient["Height"], color = "pink", s = 700, alpha = 1, edgecolor = "red", linewidth=2)
#plt.scatter(patient["Weight"], patient["Height"], marker = "x", color = "red", s = 250, alpha = 1, linewidth=5)
plt.title("Weight vs Height for Males")
plt.xlabel("Weight (lbs)")
plt.ylabel("Height (in)")
plt.axvline(x = male_sample["Weight"].median(), color = "black", linestyle = "--")
plt.axhline(y = male_sample["Height"].median(), color = "black", linestyle = "--")


plt.annotate(text = "Median Weight ({round(median_weight)} lbs)", 
             xy = (median_weight,min_height), 
             xytext = (10,-10), 
             textcoords = "offset pixels", 
             fontsize = 16)

plt.annotate(text = "Median Height ({round(median_height)} in)", 
             xy = (min_weight, median_height), 
             xytext = (-10,10), 
             textcoords = "offset pixels", 
             fontsize = 16)

plt.annotate(text = "Patient 705",
             xy = (patient["Weight"], patient["Height"]),
             xytext = (140,74),
             fontsize = 25,
             bbox = dict(boxstyle = "round",
                         fc = "salmon",
                         ec = "red"),
             arrowprops = dict(arrowstyle = "wedge, tail_width=1.",
                               fc = "salmon",
                               ec = "red",
                               patchA = None,
                               connectionstyle = "arc3,rad=-0.1"))

plt.tight_layout()
plt.savefig(fname = "C:\Home\Data_Science_Infinity\python_fundamentals\exported_plot.png")
plt.show()

#----------------------------------
#


