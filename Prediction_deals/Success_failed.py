#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:10:13 2018

@author: romain
"""

import matplotlib.pyplot as plt
import pandas as pd
from Predictions_success_failed import Predictions_success_failed


# Fonction qui calcule la moyenne que l'on met pour win un deal
def success_mean(df):
    
    print('############# SUCCESS POURCENTAGE ###########################', '\n')
    
    # On calcul la moyenne 
    mean_success = (len(df[df['status_deal'] == "success"])/len(df))*100
    
    # On affiche le resultat 
    print('Proportions des success = ', round(mean_success,2), '% \n')
    
    return mean_success

def success_courbe(df): 
    

    # On visualise l'histogramme 
    plt.figure('Proportions des success/failed')
    plt.plot(df['status_deal'])
    plt.ylabel("Nombre de deals")
    plt.title('Proportions des success/failed')    
    plt.show()
    
    
# On créer un prédicteur pour les success
def success_predicteur(df):    

    # On recupère les prédictions
    pred = Predictions_success_failed(df)
    
    # On rajoute la colonne prediction
    df['success_pred'] = pred

    return df

    
def volume_prediction_success(df):
    
     # On selectionne les inquiry qui ont success 
    data_success = df[df['success_pred'] == "success"]
    
    # On remplace nos success par des 1 pour les compter 
    data_success['success_pred'].replace(to_replace=['success'],value=1,inplace=True)
    
    #On selectionne la colonne pour les dates de payment 
    date = data_success.paid_at
    
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
    volume_prediction_success = [df_janvier_2018['status'].sum(),
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
    

    return volume_prediction_success 
    
    
    