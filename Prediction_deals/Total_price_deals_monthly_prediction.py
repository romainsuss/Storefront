#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 18:38:58 2018

@author: romain
"""

import pandas as pd

from Time_to_win import payment_date


def total_price_deals_monthly_prediction(predictions_deals, time_to_win, inquiry_data):
    
    # Données sur lesquelles on test 
    data = inquiry_data[19404:] # Ici on rentre à la main.. On peut mieux faire plus tard 
    
    # On selectionne la colonne pour le temps 
    date = inquiry_data.inquiry_created_at
    print('Date de premiere création prédite : ' , date[19404] , '\n' )
    
    # On rajoute une colonne pour nos predictions 
    data['predictions'] = predictions_deals
    
    # On selectionne les inquiry qui ont success  
    df = data[data['predictions'] == "success"]
    

    
    ######################################################
    
    #On prédit la date de payment
    jours_payment = payment_date(df)
    
    # On rajoute la colonne date de payment a notre dataframe 
    df['jours_payment'] = jours_payment
    
    
    ######################################################
    
    # On recupère le prix des jours de payment par mois 
    
    # On créer un dataframe
    jours_payment = df.jours_payment.dropna()
    prix = inquiry_data.total_price
    df_win = pd.DataFrame({'date': jours_payment, 'prix': prix[5394:]})
    
    # On enlève les valeurs NaN
    df_win = df_win.dropna()
       
    #On gère le format date
    df_win['datetime'] = pd.to_datetime(df_win['date'])
    df_win = df_win.set_index('datetime')
    df_win.drop(['date'], axis=1, inplace=True)
    
    
    # On recupère les mois 
    # Plus tard faire une fonction pour la scalabilité ...
    df_janvier_2018 = df_win[(df_win.index.month == 1) & (df_win.index.year == 2018) ]
    df_fevrier_2018 = df_win[(df_win.index.month == 2) & (df_win.index.year == 2018) ]
    df_mars_2018 = df_win[(df_win.index.month == 3) & (df_win.index.year == 2018) ]
    df_avril_2018 = df_win[(df_win.index.month == 4) & (df_win.index.year == 2018) ]
    df_mai_2018 = df_win[(df_win.index.month == 5) & (df_win.index.year == 2018) ]
    df_juin_2018 = df_win[(df_win.index.month == 6) & (df_win.index.year == 2018) ]
    df_juillet_2018 = df_win[(df_win.index.month == 7) & (df_win.index.year == 2018) ]
    df_aout_2018 = df_win[(df_win.index.month == 8) & (df_win.index.year == 2018) ]
    df_septembre_2018 = df_win[(df_win.index.month == 9) & (df_win.index.year == 2018) ]
    df_octobre_2018 = df_win[(df_win.index.month == 10) & (df_win.index.year == 2018) ]
    df_novembre_2018 = df_win[(df_win.index.month == 11) & (df_win.index.year == 2018) ]
    df_decembre_2018 = df_win[(df_win.index.month == 12) & (df_win.index.year == 2018) ]
    
    # On fait la somme des prix des deals par mois et on les met dans une liste 
    total_price_deals_monthly_prediction = [df_janvier_2018['prix'].sum(),
    df_fevrier_2018['prix'].sum(),
    df_mars_2018['prix'].sum(),
    df_avril_2018['prix'].sum(),
    df_mai_2018['prix'].sum(),
    df_juin_2018['prix'].sum(),
    df_juillet_2018['prix'].sum(),
    df_aout_2018['prix'].sum(),
    df_septembre_2018['prix'].sum(),
    df_octobre_2018['prix'].sum(),
    df_novembre_2018['prix'].sum(),
    df_decembre_2018['prix'].sum()]

    return total_price_deals_monthly_prediction
