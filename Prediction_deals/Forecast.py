#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:54:13 2018

@author: romain
"""

import pandas as pd
from datetime import timedelta 
from Predictions_time_to_win import time_to_win_pred_forecast 
from Predictions_success_failed import Predictions_success_failed



# On créer un prédicteur pour les success et le time to win 
def forecast(df):
    
    # On recupère les prédictions
    pred = Predictions_success_failed(df)
    
    # On rajoute la colonne prediction
    df['success_pred'] = pred
    
####################################################################@
    
    # On gere le format date 
    df['datetime_booked_at'] = pd.to_datetime(df['paid_at'])
    df.drop(['paid_at'], axis=1, inplace=True)
    df['datetime_created_at'] = pd.to_datetime(df['inquiry_created_at'])
    df.drop(['inquiry_created_at'], axis=1, inplace=True)
    
    
    # On trouve le time to win 
    time_to_win_deals = df['datetime_booked_at'] - df['datetime_created_at']
    df['time_to_win'] = time_to_win_deals.dt.days

    # On selectionne les success 
    X_success_pred = df[df['success_pred'] == 'success']
    X_success = df[df['status_deal'] == 'success']
    
    # On recupère les prédictions  
    pred = time_to_win_pred_forecast(X_success,X_success_pred)
    
    # On transforme nos predictions en entiers et en date  
    pred_int = list()
    for i in pred:
        pred_int.append(timedelta(int(i)))
    
    # On rajoute la colonne prediction
    X_success_pred['time_to_win_pred'] = pred_int
    
    # On rajoute une colonne pour le jour de payment prédit 
    X_success_pred['jours_payment'] = X_success_pred['datetime_created_at'] + X_success_pred['time_to_win_pred']    
    
    return X_success_pred




def volume_prediction(df):
    
     # On selectionne les inquiry qui ont success 
    data_success = df[df['success_pred'] == "success"]
    
    # On remplace nos success par des 1 pour les compter
    data_success['success_pred'].replace(to_replace=['success'],value=1,inplace=True)
    
    #On selectionne la colonne pour les dates de payment prédites
    date = data_success.jours_payment
    
    #On selectionne la colonne pour le status prédit 
    deal = data_success.success_pred
    
    #On créer un dataframe
    df = pd.DataFrame({'date': date, 'status': deal})
    
    # On gere le format date 
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')
    df.drop(['date'], axis=1, inplace=True)
     
     
    
    # On recupère les mois 
    # Plus tard faire une fonction pour la scalabilité ...
    df_janvier_2018 = df[(df.index.month == 1) & (df.index.year == 2018) ]
    df_fevrier_2018 = df[(df.index.month == 2) & (df.index.year == 2018) ]
    df_mars_2018 = df[(df.index.month == 3) & (df.index.year == 2018) ]
    df_avril_2018 = df[(df.index.month == 4) & (df.index.year == 2018) ]
    df_mai_2018 = df[(df.index.month == 5) & (df.index.year == 2018) ]
    df_juin_2018 = df[(df.index.month == 6) & (df.index.year == 2018) ]
    df_juillet_2018 = df[(df.index.month == 7) & (df.index.year == 2018) ]
    df_aout_2018 = df[(df.index.month == 8) & (df.index.year == 2018) ]
    df_septembre_2018 = df[(df.index.month == 9) & (df.index.year == 2018) ]
    df_octobre_2018 = df[(df.index.month == 10) & (df.index.year == 2018) ]
    df_novembre_2018 = df[(df.index.month == 11) & (df.index.year == 2018) ]
    df_decembre_2018 = df[(df.index.month == 12) & (df.index.year == 2018) ]
    
    #print(df_septembre_2018['prix'].max())
    
    
    # On fait la somme des prix des deals par mois et on les met dans une liste 
    volume_prediction = [df_janvier_2018['status'].sum(),
    df_fevrier_2018['status'].sum(),
    df_mars_2018['status'].sum(),
    df_avril_2018['status'].sum(),
    df_mai_2018['status'].sum(),
    df_juin_2018['status'].sum(),
    df_juillet_2018['status'].sum(),
    df_aout_2018['status'].sum(),
    df_septembre_2018['status'].sum(),
    df_octobre_2018['status'].sum(),
    df_novembre_2018['status'].sum(),
    df_decembre_2018['status'].sum()]    

    return volume_prediction 