#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:10:51 2018

@author: romain
"""

#On importe les librairies 
import numpy as np
import matplotlib.pyplot as plt
import random 
import csv
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
from datetime import datetime

#On importe les autres fichiers 
from Total_price_deals_monthly import total_price_deals_monthly
from Visualisation import visualisation
from Algo_modélisation import Algo_gradientboosting

 

### LECTURE DU FICHIER ###
inquiry_data = pd.read_csv('inquiry_prediction_dataset.csv') #On importe le fichier csv




### SANS MODELISATION ###
#On calcul la chiffre d'affaire que l'on fait par mois avec nos deals 
total_price_deals_monthly = total_price_deals_monthly()

#Visualisation par mois 
visu = visualisation(total_price_deals_monthly)



### AVEC MODELISATION ### 

#On selectionne la prediction target
y = inquiry_data.status
#On choisit les features 
inquiry_features = ['duration', 'total_price', 'nbr_messages_by_owner', 'nbr_messages_by_renter', 'nbr_messages_by_admin' ]
X = inquiry_data[inquiry_features]
#On fait nos prédictions 
predictions = Algo_gradientboosting(X,y)
print(predictions)