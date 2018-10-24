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
from Total_price_deals_monthly import total_price_deals_monthly_booked
from Visualisation import visualisation_1_courbe , visualisation_2_courbe
from Algo_modélisation import Algo_gradientboosting , Algo_randomforest
from Total_price_deals_monthly_prediction import total_price_deals_monthly_prediction
from Time_to_win import time_to_win , time_to_win_courbe

### LECTURE DU FICHIER ###
inquiry_data = pd.read_csv('pfe_inquiry_prediction.csv') #On importe le fichier csv

### SANS MODELISATION ###

#On calcul la chiffre d'affaire que l'on fait par mois avec nos deals 
total_price_deals_monthly = total_price_deals_monthly_booked()
#Visualisation par mois 
visualisation_1_courbe(total_price_deals_monthly, 'Price deals monthly no predictions 2018')

### TEMPS POUR GAGNER UN DEAL

time_to_win_deal = time_to_win(inquiry_data)
time_to_win_courbe(inquiry_data)

### AVEC MODELISATION ### 

#On selectionne la prediction target
y = inquiry_data.status
#On choisit les features 
inquiry_features = ['duration', 'total_price', 'nbr_messages_by_owner', 'nbr_messages_by_renter', 'nbr_messages_by_admin' ]
X = inquiry_data[inquiry_features]
#On fait nos prédictions 
predictions = Algo_randomforest(X,y)
#On calcul la chiffre d'affaire prédit que l'on fait par mois avec nos deals 
total_price_deals_monthly_prediction = total_price_deals_monthly_prediction(predictions, time_to_win_deal)
#Visualisation par mois
visualisation_1_courbe(total_price_deals_monthly_prediction, 'Price deals monthly predictions 2018')


### VISUALISATION POUR COMPARER ### 

visualisation_2_courbe(total_price_deals_monthly,total_price_deals_monthly_prediction)

