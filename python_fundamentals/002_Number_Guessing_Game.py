# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:32:52 2022

@author: jampa
"""

#CORRECT_NUMBER = 26

#user_guess = input("what is your guess ?")

#print(user_guess)

import random

#random.randint(1,10)

LOWER_BOUND = 1
UPPER_BOUND = 100
GUESS_LIMIT = 5
GUESS_COUNTER = 0
CORRECT_NUMBER = random.randint(LOWER_BOUND,UPPER_BOUND)

print(f"Try guessing a number between {LOWER_BOUND} and {UPPER_BOUND}. You have {GUESS_LIMIT} guesses ")

while True:
    user_guess = int(input("enter your guess ?"))
    GUESS_COUNTER +=1
    reminaing_guesses = GUESS_LIMIT - GUESS_COUNTER
    if LOWER_BOUND <= user_guess <= UPPER_BOUND:
        if user_guess == CORRECT_NUMBER:
            print(f"Congratulations on guessing the number. You took {GUESS_COUNTER} guesses to get correct answer")
            break
        elif user_guess > CORRECT_NUMBER:
            print(f"{user_guess} guess is greater than the correct number.You have {reminaing_guesses} guesses left.")
        else:
            print(f"{user_guess} guess is less than the correct number. You have {reminaing_guesses} guesses left.")
    else:
         print(f"entered number is out of range. You have {reminaing_guesses} guesses left")
    
    if reminaing_guesses == 0:
        print(f"Sorry, all the guesses are done. You are after number {CORRECT_NUMBER}")
        break