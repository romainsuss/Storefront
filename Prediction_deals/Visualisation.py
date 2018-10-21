#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:09:41 2018

@author: romain
"""

import numpy as np
import matplotlib.pyplot as plt
import random 
import csv
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
from datetime import datetime

def visualisation_1_courbe(y):
    
    x = np.arange(1,13)
    plt.plot(x, y)
    plt.xlabel("mois")
    plt.ylabel("total price")
    plt.title('Price_deals_monthly 2018')
    
    plt.show()
    
    
def visualisation_2_courbe(y,y_pred): 
    
    x = np.arange(1,13)
    plt.axvline(x=6,color='red')
    p1 = plt.plot(x, y, marker='o')
    p2 = plt.plot(x, y_pred, marker='v')
    plt.legend([p1, p2], ["Price_deals", "Price_deals_prédiction"])
    plt.xlabel("mois")
    plt.ylabel("total price")
    plt.title('Price_deals_monthly 2018')
    
    plt.show()