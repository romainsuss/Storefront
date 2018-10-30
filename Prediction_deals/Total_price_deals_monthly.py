#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:02:43 2018

@author: romain
"""

#test =  datetime.strptime('2018-10-31 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')
# https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea



import pandas as pd





def total_price_deals_monthly_booked(inquiry_data):
    
    print('\n', '################ TOTAL PRICE DEALS MONTHLY SANS PREDICTION #########')
   
    # On selectionne les inquiry qui ont success 
    data_success = inquiry_data[inquiry_data['status_deal'] == "success"]
    
    #On selectionne la colonne pour les inquiry paid 
    date = data_success.paid_at
    
    #On selectionne la colonne pour le prix 
    prix = data_success.total_price
    
    #On créer un dataframe
    df = pd.DataFrame({'date': date, 'prix': prix})
    
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
    
    
    # On fait la somme des prix des deals par mois et on les met dans une liste 
    total_price_deals_monthly_booked = [df_janvier_2018['prix'].sum(),
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
    

    return total_price_deals_monthly_booked

