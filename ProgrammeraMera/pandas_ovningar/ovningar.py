# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 14:32:26 2021

@author: danie
"""
import pandas as pd

#data = pd.read_csv('Automobile_data.csv')
#print('Fem första raderna')
#print(data.head())
#print('Fem sista raderna')
#print(data.tail())

df = pd.read_csv("Automobile_data.csv", na_values={
'price':["?","n.a"],
'stroke':["?","n.a"],
'horsepower':["?","n.a"],
'peak-rpm':["?","n.a"],
'average-mileage':["?","n.a"]})

df.to_csv("Automobile_data.csv", index=False)

df = df.sort_values(by='price', ascending=False)
print (df)
#most_expensive = df.head(1)
#print(most_expensive[['company', 'price']])

df.groupby()