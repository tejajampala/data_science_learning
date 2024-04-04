
"""
Created on Wed May 25 14:45:04 2022

@author: jampa
"""

#string functions
my_string="good-boy"

my_string.index("d")
my_string.replace("o", "a")
my_string.upper()
my_string.lower()
my_string.title()
my_string.split("-")
my_string
my_string.count("o")

#--------------------

#number functions
a = 10
b = 20
c = 1535.4566

max(a,b)
min(a,b)
abs(a)
round(a,3)
int(a)
round(c,-2)
round(c,-3)

#-----------------------

#list functions

my_list = list()
my_list = [5, 6, 7, 8]
my_list

my_list.append(9)
my_list.insert(5,10)
my_list.extend([11,12,13])
my_list = [1,2,3,4] + my_list
my_list.remove(13)
del my_list[11]
my_list
my_list.pop(10)
my_list.index(10)
my_list.sort()
my_list.sort(reverse=True)

my_list2 = my_list.copy()

my_list.pop(10)

#------------------------

#Tuples
# we cannot add or remove elements
#immutable

my_tuple = (1,"4",True)
my_tuple

my_tuple.index("4")

#----------------

#sets
#No duplicates
my_set = set()
my_set = {1,2}
my_set2 = {3,4,5,6}

my_set.add(3)
my_set.update({4,5})

my_set.discard(7) # does not throw an error
#my_set.remove(9) # throws an error

my_set.difference(my_set2)
#my_set2.difference(my_set)
my_set.difference_update(my_set2)  #myset is updated

my_set.intersection(my_set2)
my_set.intersection_update(my_set2)

test_set = {1,2,3,4}
test_set.pop() #can pop any element from the set

sorted(test_set)

#----------------------

#dict
my_dict = dict()

my_dict = {1:"one", 2:"Two"}

my_dict.get(3) #does not throw an error

1 in my_dict

my_dict.keys()
my_dict.values()

my_dict.pop(2)
my_dict

#gets key value pair
my_dict.items()

#-----------------
#conditional statements

if True:
    print("Heloo if")
elif False:
    print("yes")
else:
    print("in else")    
    
#-----------------
#For loops

#enumerate gets the index as well

my_list_for = ["a", "b", "c"]

for idx,i in enumerate(my_list_for):
    print(idx,i)
    
#range(start, stop, step)

for i in range(10,100,10):
    print(i)

#continue statment
#break

i = 1
while i<=5:
    print(i)
    i += 1
    
  
#input

fav_num = input("wat is your fav num?")

#--------------
# functions
def happy_birthday():
    pass

def happy_birthday(name="Human"):
    print(f"Happy Birthday {name}")

happy_birthday("teja")

#--------------------
# list comprehension
my_list = [1,2,3,4,5]
my_list1 = []


my_list1 = [i for i in my_list]

my_list1 = [i*i for i in my_list]

my_list1 = [i*i for i in my_list if i % 2 == 0] 

my_list2 = [[3,4], [4,5], [8,9]]

my_list3 = [i[0] for i in my_list2]


my_list4 = [i[0] for i in my_list2 if i[1] == max(i[1] for i in my_list2)]

#------------
# exception handling

try:
    age = int(input("what is my age ?"))
    print(age)
except:
    print("not a valid input, please try again")
    
print(some_object) #name error

my_list = [1,2]
abmy_list[4] #index out of range

"a"+5 #ValueError

int("python") #TypeError

3/0 #zero division error



