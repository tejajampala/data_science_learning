# -*- coding: utf-8 -*-
"""
Created on Fri May 27 23:38:17 2022

@author: jampa
"""

#Numpy is written in C
#Numpy is more efficient with memory usage
#Numpy can run sub-tasks in parallel
#Numpy is high performance multi dimensional array

#Liner Algebra & Image Processing and Computer Vision

#Numpy allows only single type of elements

import numpy as np

my_id_array = np.array([1,2,3])
type(my_id_array)

my_id_array.shape # (3,)

my_id_array[0]

my_id_array[0:2]

my_2d_array = np.array([[1,2,3,4,5],[6,7,8,9,10]])

my_2d_array.shape

my_2d_array[0][1]
my_2d_array[0,1]

my_2d_array[0:2,1:3]

np.zeros(3)
np.zeros((3,3))
np.zeros((3,3,3))
np.ones((3,3,3))

np.full((3,3), 5) #fills values with 5


np.arange(10) #rane of elements
np.arange(2,10)
np.arange(2,10,2)

np.linspace(1,5,20) #evenly 20 spaced points from 1 to 5

float_array = np.linspace(1,5,20)
np.round(float_array,2)

np.random.rand(5)
test = np.random.rand(5,2) # 5,2 array
test

test.shape

np.random.randint(20,80,100) # numbers from 20 to 80 by 100
np.random.randint(20,80,(10,10)) #2d array

my_id_array = np.random.randint(20,80,100)

#reshape
my_2d_array = my_id_array.reshape(10,10) #convert 1d to 2d
print(my_2d_array)

#------------------------
#arithmetic calculations
my_id_array = np.random.randint(2,8,16)

my_id_array.max()
my_id_array.min()
my_id_array.mean()
my_id_array.sum()
my_id_array.std()

#convert 1d to 2d
my_2d_array = my_id_array.reshape(4,4)
print(my_2d_array)
my_2d_array.max()
my_2d_array.max(axis=0)
my_2d_array.max(axis=1)

#to get the index of max & min in the array
my_2d_array.argmax(axis=0)
my_2d_array.argmax(axis=1)

np.sort(my_id_array)


a = np.array([1,2,3,4,5])

a+10
a/10
a-10
a*10


b= np.array([-2,-1,0,1,2])

np.square(a)
np.sqrt(a)
np.sign(a)


a = np.array([1,2,3])
b = np.array([4,5,6])

np.dot(a,b)

#--------------------
#updating and finding values

my_id_array = np.zeros(10)
print(my_id_array)

my_id_array[0] = 50
print(my_id_array)

my_id_array[3:6] = 50

np.where(my_id_array) #checks for non-zero values
np.where(my_id_array == 50)

my_2d_array = np.array([[1,5,9],[8,5,5]])
print(my_2d_array)

np.where(my_2d_array == 50)
np.where(my_2d_array > 5)
np.where(my_2d_array < 5)
np.where(my_2d_array != 5)

np.argwhere(my_2d_array == 5)


index = np.where(my_2d_array > 5)
my_2d_array[index]
my_2d_array[index] = 100
print(my_2d_array)

#----------------------
np.all(my_id_array) #checks for non-zero values and returns a boolean value
np.all(my_id_array >=0)
np.all(my_id_array > 5)

np.any(my_id_array) #checks for any non-zero value

#stack arrays - vstack(vertical) and hsctak (horizontal)
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

v = np.vstack((a,b)) #number of columns need to match
print(v)

v = np.vstack((a,b,a,b))
print(v)

h = np.hstack((a,b)) #number of rows need to match
print(h)

h = np.hstack((a,b,a,b))
print(h)

#flatten
print(my_2d_array)
my_2d_array.flatten()










