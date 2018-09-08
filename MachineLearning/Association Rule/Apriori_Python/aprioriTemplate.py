# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 07:52:00 2018

@author: Harry
"""

#A priori

#Importing the libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dt = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

#The algorithm requires a list of lists of all the transactions
#so we need to transform the dataframe
transactions = []
for i in range(0,7501):
    transactions.append([str(dt.values[i,j]) for j in range(0,20)])
    #transactions.append([str(dt.values[i])])
    
#Training Apriori on the dataset
from apyori import apriori
#The keyword sdepend largely from the business problem
#so we have to experiment a priori with the minimum vlaues
#until we find satisfying rules.
#In this case for min_support we assume a product is bought 3 times
#a day for 7 days => 7*3/7500 = 0,0028
#For min_confidence we assume that 20% of the prods
#are bought together
#For lift we try lift = 3
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, 
                min_lift = 3, min_length = 2)

#Visualising the results
results = list(rules)
results_list = []
for i in range(0, len(results)): 
    results_list.append('RULE:\t' + str(results[i][0]) +
                        '\nSUPPORT:\t' + str(results[i][1]) +
                        '\nCONFIDENCE:\t' + str(results[i][2][0][2]) +
                        '\nLIFT:\t' + str(results[i][2][0][3]))