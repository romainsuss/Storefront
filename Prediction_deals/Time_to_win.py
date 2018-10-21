#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:00:41 2018

@author: romain
"""

import numpy as np
import matplotlib.pyplot as plt
import random 
import csv
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns


### LECTURE DU FICHIER ###
inquiry_data = pd.read_csv('inquiry_prediction_dataset.csv') #On importe le fichier csv 

# On regarde les 5 premières lignes de notre fichier qui contient les data 
# Cela nous permet d'avoir une première vu de ce qu'il contient 
inquiry_data.head()



# les variables dont on a besoin 
created_at = inquiry_data['created_at']
#payment_at = inquiry_data['payment_at']


def time_to_win(created_at, payment_at):
    
    time_to_win = 7 
    return time_to_win 