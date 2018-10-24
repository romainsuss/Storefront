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
inquiry_data = pd.read_csv('pfe_inquiry_prediction.csv') #On importe le fichier csv 




def time_to_win(df):
    
    print('############# TIME TO WIN A DEAL ###########################', '\n')
          
    # On selectionne les colonnes sans NaN, celle qui ont eu un payment 
    df = inquiry_data.dropna()
    
    #On selectionne la colonne pour la date de création 
    date_creation = df.created_at
    
    #On selectionne la colonne pour la date de payment 
    date_payment = df.booked_at
    
    #On créer un dataframe
    df_dates = pd.DataFrame({'created_at': date_creation , 'booked_at': date_payment})
    
    #On gère le format date
    df_dates['datetime_booked_at'] = pd.to_datetime(df_dates['booked_at'])
    df_dates.drop(['booked_at'], axis=1, inplace=True)
    
    df_dates['datetime_created_at'] = pd.to_datetime(df_dates['created_at'])
    df_dates.drop(['created_at'], axis=1, inplace=True)
    
    time_to_win_deals = df_dates['datetime_booked_at'] - df_dates['datetime_created_at']
    
    mean_time_to_win_deals = time_to_win_deals.mean()
    
    print('Nombre de jours moyen pour gagner un deal = ', mean_time_to_win_deals.days, '\n')
    
    return mean_time_to_win_deals.days





def time_to_win_courbe(df):
        # On selectionne les colonnes sans NaN, celle qui ont eu un payment 
    df = inquiry_data.dropna()
    
    #On selectionne la colonne pour la date de création 
    date_creation = df.created_at
    
    #On selectionne la colonne pour la date de payment 
    date_payment = df.booked_at
    
    #On créer un dataframe
    df_dates = pd.DataFrame({'created_at': date_creation , 'booked_at': date_payment})
    
    #On gère le format date
    df_dates['datetime_booked_at'] = pd.to_datetime(df_dates['booked_at'])
    df_dates.drop(['booked_at'], axis=1, inplace=True)
    
    df_dates['datetime_created_at'] = pd.to_datetime(df_dates['created_at'])
    df_dates.drop(['created_at'], axis=1, inplace=True)
    
    time_to_win_deals = df_dates['datetime_booked_at'] - df_dates['datetime_created_at']
    
    #On creer une liste avec tous les nombre de jours 
    liste_day = list()
    for time in time_to_win_deals:
        liste_day.append(time.days)
    
    #On visualise 
    plt.hist(liste_day,bins=50)
    plt.xlabel("jours")
    plt.ylabel("total jours")
    plt.title('Nombre de jours pour gagner un deal')    
    plt.show()
    








