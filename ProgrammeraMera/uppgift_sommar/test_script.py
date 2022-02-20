# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 16:42:48 2021

@author: danie
"""
import numpy as np
import uppgift_prgrmera as up
import pandas as pd


def test_2e_3k():
    medel_std_matris = [[1, 5, 1, 4, 2], [2, 4, 2, 6, 3], [3, 3, 2, 10, 1]]
     
    test_data = up.kNN_data(20, medel_std_matris)
    okanda_objekt = np.array([[10, 6], [3, 2], [4, 4], [6, 8]])
    k = 7
    [klassifieringar, avstand] = up.kNN_algoritm(k, test_data, okanda_objekt)
    up.kNN_analysera(k, test_data, okanda_objekt, avstand, klassifieringar, korrekt_resultat=[[1]])
    
    
def test_1_okand():
     medel_std_matris = [[1, 5, 1, 4, 2], [2, 4, 2, 6, 3], [3, 3, 2, 10, 1]]
     test_data = up.kNN_data(20, medel_std_matris)
     okand = [[5.6, 2.4]]
     k = 7
     [klassifieringar, avstand] = up.kNN_algoritm(k, test_data, okand)
     
     up.kNN_analysera(k, test_data, okand, avstand[0:k], klassifieringar)

def test_3e_3k():
    medel_std_matris = [[1, 5, 1, 4, 2, 4, 5], [2, 4, 2, 6, 3, 10, 5], [3, 3, 2, 10, 1, 8, 10]]
     
    test_data = up.kNN_data(20, medel_std_matris)
    #okanda_objekt = np.array([[10, 6, 5], [3, 2, 4], [4, 4, 10], [6, 8, 9]])
    okanda_objekt = [10, 10, 5]
    k = 7
    [klassifieringar, avstand] = up.kNN_algoritm(k, test_data, okanda_objekt)
    up.kNN_analysera(k, test_data, okanda_objekt, avstand, klassifieringar, korrekt_resultat=[[1]])
    
    return 0

def test_2e_2k():
    
    medel_std_matris = [[1, 5, 1, 4, 2], [2, 4, 2, 6, 3]]
     
    test_data = up.kNN_data(20, medel_std_matris)
    #okanda_objekt = np.array([[10, 6, 5], [3, 2, 4], [4, 4, 10], [6, 8, 9]])
    okanda_objekt = [5, 5]
    k = 7
    [klassifieringar, avstand] = up.kNN_algoritm(k, test_data, okanda_objekt)
    up.kNN_analysera(k, test_data, okanda_objekt, avstand, klassifieringar)
    
    return 0

def test_fil():
    
    traning_data = pd.read_csv('traningsdata_2klasser_2egenskaper.csv', header=None)
    korrekt_resultat = pd.read_csv('korrekt_klassificering_2klasser_2egenskaper.csv', header=None)
    okanda_objekt = pd.read_csv('okanda_objekt_2klasser_2egenskaper.csv', header=None)
    traning_data = traning_data.to_numpy()
    korrekt_resultat = korrekt_resultat.to_numpy()
    okanda_objekt = okanda_objekt.to_numpy()
    
    k = 10
    kNN_resultat = up.kNN_algoritm(k, traning_data, okanda_objekt)
    up.kNN_analysera(k, traning_data, okanda_objekt, None, kNN_resultat[0], korrekt_resultat)
    
    return 0

#test_1_okand()    
test_2e_3k()
#test_3e_3k()    
#test_2e_2k()   
#test_fil()





