# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:36:24 2021

@author: danie
"""

import math

class Film:
    
    def __init__(self, title, director, length, main_character, price):
        self.title = title
        self.director = director
        self.length = length
        self.main_character = main_character
        self.price = price
        
    def __repr__(self):
        return f"""Title: {self.title}\nDirector: {self.director}\nLength: {self.length} min\n\
Main character: {self.main_character} \nPrice: {self.price}\n"""
    
    def set_price(self, price):
        
        if price < self.price*0.9:
            print('För lågt pris sätts')
        elif price > self.price*1.1:
            print('Priset höjs för mycket, inget händer')
        else:
            self.price= price
            print(f'Priset ändrades till: {self.price}')
    
class Person:
    
    def __init__(self, first_name='', last_name='', yob=-1, height=-1):
        self.first_name = first_name
        self.last_name = last_name
        self.yob = yob
        self.height = height
        
    def __repr__(self):
        return f'First name: {self.first_name}\nLast name: {self.last_name}\nYear of birth: {self.yob}\nHeight: {self.height} cm'
    
    
def test():
    
    film1 = Film('GBU', Person('Daniel', 'Ringwald', 1999, 180), 124, 'Leo', 123)
    film2 = Film('swag', Person('Olof', 'Brandt', 1999, 185), 224, 'Klas', 13)
    
    print(film1)
    print(film2)
    
