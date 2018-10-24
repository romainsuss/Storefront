#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 18:38:58 2018

@author: romain
"""

import numpy as np
import matplotlib.pyplot as plt
import random 
import csv
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
from datetime import datetime, timedelta 

from Time_to_win import time_to_win

### LECTURE DU FICHIER ###
inquiry_data = pd.read_csv('pfe_inquiry_prediction.csv') #On importe le fichier csv 

# On regarde notre fichier qui contient les data 
# Cela nous permet d'avoir une première vu de ce qu'il contient 
#print(inquiry_data)

def total_price_deals_monthly_prediction(predictions, time_to_win):
    

    
    #On selectionne la colonne pour le temps 
    date = inquiry_data.created_at
    print('Date de premiere création prédite : ' , date[5394] , '\n' )
    
    #On selectionne la colonne pour le prix 
    prix = inquiry_data.total_price
    
    #On créer un dataframe
    # On rajoute au dataframe initial une colonne avec nos prédictions sur les X
    df_pred = pd.DataFrame({'date': date[5394:], 'prix': prix[5394:], 'prediction' : predictions})
    
    # On selectionne les inquiry qui ont success  
    df = df_pred[df_pred['prediction'] == "success"]
    
    #On gère le format date
    df['datetime'] = pd.to_datetime(df['date'])
    #df = df.set_index('datetime')
    df.drop(['date'], axis=1, inplace=True)
    
    ######################################################

    #On rajoute les jours pour gagner le deal     
    date_win  = df['datetime'] + timedelta(time_to_win)
    
    #On créer un dataframe
    df2 = pd.DataFrame({'date': date_win, 'prix': prix[5394:], 'prediction' : predictions})
    
    # On selectionne les inquiry qui ont success
    df_win = df2[df2['prediction'] == "success"]
    print('nombre de success predit =' , len(df_win), '\n')
    
    #On gère le format date
    df_win['datetime'] = pd.to_datetime(df_win['date'])
    df_win = df_win.set_index('datetime')
    df_win.drop(['date'], axis=1, inplace=True)
    
    
    
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
