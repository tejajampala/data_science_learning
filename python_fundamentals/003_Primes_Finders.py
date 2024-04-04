# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:35:07 2022

@author: jampa
"""

def primes_finder(n):
    #number range to be checked
    number_range = set(range(2,n+1))
    
    #empty_list
    primes_list = []
    
    #while loop
    while number_range:
        prime = min(sorted(number_range))
        number_range.remove(prime)
        primes_list.append(prime)
        multiples = set(range(prime*2, n+1, prime))
        number_range.difference_update(multiples)
    
    prime_count = len(primes_list)
    
    largest_prime = max(primes_list)
    
    print(f"There are {prime_count} prime numbers between 2 and {n}, the largest of which is {largest_prime}")
    
    return primes_list

primes_list = primes_finder(100)
