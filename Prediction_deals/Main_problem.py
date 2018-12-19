#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:10:51 2018

@author: romain
"""

#On importe les librairies 
import pandas as pd

#On importe les autres fichiers 
from Total_price_deals_monthly import total_price_deals_monthly_booked, volume_deals_monthly_booked
from Visualisation import visualisation_1_courbe , visualisation_2_courbe
from Time_to_win import time_to_win_mean, time_to_win_courbe , time_to_win_predicteur, total_price_prediction_time_to_win
from Success_failed import success_mean, success_courbe, success_predicteur, volume_prediction_success
from Forecast import forecast , volume_prediction


#---------------------------------------------------------------------------#
                    # LECTURE DU FICHIER #
#---------------------------------------------------------------------------#
  
# On lit notre fichier                   
df = pd.read_csv('inquiry.csv' , delimiter=';')
# On gère le format date     
df['datetime_created_at'] = pd.to_datetime(df['inquiry_created_at'])
df.drop(['inquiry_created_at'], axis=1, inplace=True)
# On trie les dates created_at
data_sort = df.sort_values(by=['datetime_created_at'])
# On renome la colonne pour quelle garde le nom initial
inquiry_data = data_sort.rename(columns={"datetime_created_at": "inquiry_created_at"})
# On enlève la dernière ligne
inquiry_data = inquiry_data.drop([20452])

#inquiry data [20452 rows x 19 columns]


#---------------------------------------------------------------------------#
                    # DATASET SUCCESS #
#---------------------------------------------------------------------------#
 
# On selectionne les success 
X_success = inquiry_data[inquiry_data['status_deal'] == 'success']
#On gère le format date
X_success['datetime_booked_at'] = pd.to_datetime(X_success['paid_at'])
X_success.drop(['paid_at'], axis=1, inplace=True)
X_success['datetime_created_at'] = pd.to_datetime(X_success['inquiry_created_at'])
X_success.drop(['inquiry_created_at'], axis=1, inplace=True)

# On trouve le time to win 
time_to_win_deals = X_success['datetime_booked_at'] - X_success['datetime_created_at']
X_success['time_to_win'] = time_to_win_deals.dt.days

# success [901 rows x 20 columns]


#---------------------------------------------------------------------------#
                    # DEALS SUCCESS #
#---------------------------------------------------------------------------#

# On calcul le volume de deals success par mois  
#total_price_deals_monthly = total_price_deals_monthly_booked(inquiry_data)
volume_price_deals_monthly = volume_deals_monthly_booked(inquiry_data)
        
# Visualisation par mois 
visualisation_1_courbe(volume_price_deals_monthly, 'Volume Price deals monthly 2018')

#---------------------------------------------------------------------------#
                    # TIME TO WIN #
#---------------------------------------------------------------------------#

# On regarde la moyenne des time to win 
moyenne_time_to_win = time_to_win_mean(X_success)
# On trace l'histogramme de notre time to win 
time_to_win_courbe(X_success)
# On prédit le time to win avec un prédicteur 
df_time_to_win_deal = time_to_win_predicteur(X_success)

# On regarde notre resultat juste avec le prédicteur time to win 
total_price_prediction_time_to_win = total_price_prediction_time_to_win(df_time_to_win_deal)
visualisation_2_courbe(volume_price_deals_monthly,total_price_prediction_time_to_win,'Prédiction time to win')



#---------------------------------------------------------------------------#
                    # SUCCESS/FAILED  #
#---------------------------------------------------------------------------#

# On regarde le pourcentage des success 
moyenne_success = success_mean(inquiry_data)
# On trace l'histogramme des success/failed 
#success_courbe(inquiry_data)
# On prédit le time to win avec un prédicteur 
df_success_pred = success_predicteur(inquiry_data)
# On regarde notre resultat juste avec le prédicteur time to win 
volume_prediction_success = volume_prediction_success(df_success_pred)
visualisation_2_courbe(volume_price_deals_monthly,volume_prediction_success,'Prédictions des success')


#---------------------------------------------------------------------------#
                    # TIME TO WIN + SUCCESS/FAILED  #
#---------------------------------------------------------------------------#

df_forecast = forecast(inquiry_data)

## On regarde notre resultat juste avec le prédicteur time to win 
volume_prediction = volume_prediction(df_forecast)
visualisation_2_courbe(volume_price_deals_monthly,volume_prediction,'Forecast')







                    



      









