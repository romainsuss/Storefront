#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:10:51 2018

@author: romain
"""

#On importe les librairies 
import pandas as pd


#On importe les autres fichiers 
from Total_price_deals_monthly import total_price_deals_monthly_booked
from Visualisation import visualisation_1_courbe , visualisation_2_courbe
from Algo_modélisation import Algo_randomforest_classification
from Algo_modélisation_monthly import Monthly_performance
from Total_price_deals_monthly_prediction import total_price_deals_monthly_prediction
from Time_to_win import time_to_win_courbe , time_to_win_predicteur


#---------------------------------------------------------------------------#
                    # LECTURE DU FICHIER #
#---------------------------------------------------------------------------#
                    
df = pd.read_csv('inquiry.csv' , delimiter=';')
# On gere le format date     
df['datetime_created_at'] = pd.to_datetime(df['inquiry_created_at'])
df.drop(['inquiry_created_at'], axis=1, inplace=True)
# On trie les dates created_at
data_sort = df.sort_values(by=['datetime_created_at'])
# On renome la colonne pour quelle garde le nom initial
inquiry_data = data_sort.rename(columns={"datetime_created_at": "inquiry_created_at"})


#---------------------------------------------------------------------------#
                    # DEALS SUCCESS SANS MODELISATION #
#---------------------------------------------------------------------------#

# On calcul la chiffre d'affaire que l'on fait par mois avec nos deals 
total_price_deals_monthly = total_price_deals_monthly_booked(inquiry_data)
# Visualisation par mois 
visualisation_1_courbe(total_price_deals_monthly, 'Price deals monthly no predictions 2018')

#---------------------------------------------------------------------------#
                    # TIME TO WIN #
#---------------------------------------------------------------------------#

# On trace l'histogramme de notre time to win 
time_to_win_courbe(inquiry_data)
# On prédit le time to win avec un prédicteur 
time_to_win_deal = time_to_win_predicteur(inquiry_data)


#---------------------------------------------------------------------------#
                    # DEALS SUCCESS AVEC MODELISATION #
#---------------------------------------------------------------------------#

# On selectionne la prediction target
y = inquiry_data.status_deal.dropna()
# On choisit les features 
inquiry_features = ['duration_event_days', 'total_price', 'main_nbr_admin_message', 'main_nbr_renter_message', 'main_nbr_renter_message.1' ]
X = inquiry_data[inquiry_features].dropna()

# On fait nos prédictions de deals success 
predictions = Algo_randomforest_classification(X,y)
# On calcul la chiffre d'affaire prédit que l'on fait par mois avec nos deals 
total_price_deals_monthly_prediction = total_price_deals_monthly_prediction(predictions, time_to_win_deal, inquiry_data)
#Visualisation par mois
visualisation_1_courbe(total_price_deals_monthly_prediction, 'Price deals monthly predictions 2018')

#---------------------------------------------------------------------------#
                    # VISUALISATION POUR COMPARER #
#---------------------------------------------------------------------------#

# On visualise les courbes réalisé vs forecast pour comparer 
visualisation_2_courbe(total_price_deals_monthly,total_price_deals_monthly_prediction)

#---------------------------------------------------------------------------#
                    # PERFORMANCE MOIS #
#---------------------------------------------------------------------------#

# On regarde notre performance par mois 
#Monthly_performance()

#---------------------------------------------------------------------------#
                    # DATA #
#---------------------------------------------------------------------------#

#print('######## Données réalisées #######')
#print('nombre de deals success sur ensemble de test = 539')
#print('nombre de deals success environ predit  = 392')
                    
                    



      









