#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:00:41 2018

@author: romain
"""


import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta 
from Predictions_time_to_win import time_to_win_predictions 


# Fonction qui calcule la moyenne que l'on met pour win un deal
def time_to_win_mean(df):
    
    print('############# TIME TO WIN A DEAL MEAN ###########################', '\n')
    
    # On calcul la moyenne 
    mean_time_to_win_deals = df['time_to_win'].mean()
    
    # On affiche le resultat 
    print('Nombre de jours moyen pour gagner un deal = ', mean_time_to_win_deals, '\n')
    
    return mean_time_to_win_deals




# On cherche à tracer l'histogramme des times to win 
def time_to_win_courbe(df): 
    
    # On visualise l'histogramme 
    plt.figure('Nombre de jours pour gagner un deal')
    plt.hist(df['time_to_win'])
    plt.xlabel("jours")
    plt.ylabel("total jours")
    plt.title('Nombre de jours pour gagner un deal')    
    plt.show()
    

# On créer un prédicteur pour le time to win 
def time_to_win_predicteur(df):
    
    # On recupère les prédictions 
    pred = time_to_win_predictions(df)
    
    # On transforme nos predictions en entiers et en date  
    pred_int = list()
    for i in pred:
        pred_int.append(timedelta(int(i)))
    
    # On rajoute la colonne prediction
    df['time_to_win_pred'] = pred_int
     
    # On rajoute une colonne pour le jour de payment prédit 
    df['jours_payment'] = df['datetime_created_at'] + df['time_to_win_pred']

    return df

def total_price_prediction_time_to_win(df):
    
    # On remplace nos success par des 1 pour les compter 
    df['status_deal'].replace(to_replace=['success'],value=1,inplace=True)
    
     #On selectionne la colonne pour les dates de payments prédites 
    date = df.jours_payment
    
    #On selectionne la colonne pour le status 
    deal = df.status_deal
    
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
    total_price_deals_monthly_prediction = [df_janvier_2018['status'].sum(),
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
    
    return total_price_deals_monthly_prediction





