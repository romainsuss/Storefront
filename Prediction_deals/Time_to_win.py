#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:00:41 2018

@author: romain
"""


import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta 
from sklearn.ensemble import RandomForestClassifier

from Algo_modélisation import Algo_randomforest_regression


# Fonction qui calcule la moyenne que l'on met pour win un deal
def time_to_win(df):
    
    print('############# TIME TO WIN A DEAL ###########################', '\n')
          
    # On selectionne les colonnes sans NaN, celle qui ont eu un payment 
    df = df.dropna()
    
    # On selectionne la colonne pour la date de création 
    date_creation = df.inquiry_created_at
    
    # On selectionne la colonne pour la date paid 
    date_payment = df.paid_at
    
    # On créer un dataframe
    df_dates = pd.DataFrame({'created_at': date_creation , 'booked_at': date_payment})
    
    # On gère le format date
    df_dates['datetime_booked_at'] = pd.to_datetime(df_dates['booked_at'])
    df_dates.drop(['booked_at'], axis=1, inplace=True)
    df_dates['datetime_created_at'] = pd.to_datetime(df_dates['created_at'])
    df_dates.drop(['created_at'], axis=1, inplace=True)
    
    # On fait la différence en la date de payment et la date de création 
    time_to_win_deals = df_dates['datetime_booked_at'] - df_dates['datetime_created_at']
    
    # On calcul la moyenne 
    mean_time_to_win_deals = time_to_win_deals.mean()
    
    # On affiche le resultat 
    print('Nombre de jours moyen pour gagner un deal = ', mean_time_to_win_deals.days, '\n')
    
    return mean_time_to_win_deals.days




# On cherche à tracer l'histogramme des times to win 
def time_to_win_courbe(df): 
    
    print('############# TIME TO WIN A DEAL HISTOGRAMME ###########################', '\n')

    
    # On selectionne la colonne pour la date de création 
    date_creation = df.inquiry_created_at
    
    # On selectionne la colonne pour la date de payment 
    date_payment = df.paid_at
    
    # On créer un dataframe
    df_dates = pd.DataFrame({'created_at': date_creation , 'booked_at': date_payment})
    
    # On selectionne les colonnes sans NaN, celle qui ont eu un payment 
    df_dates = df_dates.dropna()
    
    # On gère le format date
    df_dates['datetime_booked_at'] = pd.to_datetime(df_dates['booked_at'])
    df_dates.drop(['booked_at'], axis=1, inplace=True)
    df_dates['datetime_created_at'] = pd.to_datetime(df_dates['created_at'])
    df_dates.drop(['created_at'], axis=1, inplace=True)
    
    # On fait la différence en la date de payment et la date de création 
    time_to_win_deals = df_dates['datetime_booked_at'] - df_dates['datetime_created_at']
    
    # On creer une liste avec tous les differents time to win 
    liste_day = list()
    for time in time_to_win_deals:
        liste_day.append(time.days)

    # On visualise l'histogramme 
    plt.hist(liste_day)
    plt.xlabel("jours")
    plt.ylabel("total jours")
    plt.title('Nombre de jours pour gagner un deal')    
    plt.show()
    

# On créer un prédicteur pour le time to win 
def time_to_win_predicteur(df):
              
    #On selectionne la colonne pour la date de création 
    date_creation = df.inquiry_created_at
    
    #On selectionne la colonne pour la date de payment 
    date_payment = df.paid_at
    
    #On créer un dataframe
    df_dates = pd.DataFrame({'created_at': date_creation , 'booked_at': date_payment})
    
    # On selectionne les colonnes sans NaN, celle qui ont eu un payment 
    df_dates = df_dates.dropna()
    
    
    #On gère le format date
    df_dates['datetime_booked_at'] = pd.to_datetime(df_dates['booked_at'])
    df_dates.drop(['booked_at'], axis=1, inplace=True)
    df_dates['datetime_created_at'] = pd.to_datetime(df_dates['created_at'])
    df_dates.drop(['created_at'], axis=1, inplace=True)
    
    # On fait la différence en la date de payment et la date de création 
    time_to_win_deals = df_dates['datetime_booked_at'] - df_dates['datetime_created_at']


    # On rajoute au dataframe une derniere colonne : le time to win 
    df['time_to_win'] = time_to_win_deals
    
    # On créer notre target     
    liste_day = list()
    for time in time_to_win_deals:
        liste_day.append(time.days)
    y = liste_day
    
    # On créer notre dataset de prédiction
    inquiry_features = ['paid_at','inquiry_created_at','duration_event_days', 'total_price', 'main_nbr_admin_message', 'main_nbr_renter_message', 'main_nbr_renter_message.1' ]
    X = df[inquiry_features].dropna()
    
    # On enlève les colonnes qui ne servent pas à notre prédicteur
    del X['paid_at']
    del X['inquiry_created_at']

    # On fait notre prédiction
    pred = Algo_randomforest_regression(X,y)
    
    #pred = Algo_gradientboosting_regression(X,y)
    
    return pred



# Meme chose que pour le time to win predicteur sans la prédiction finale
# Ici on retourne juste X et y 
# On en aura besoin pour faire la prédiction sur notre ensemble de test 
def time_to_win_X_y(df):
    
    print('############# TIME TO WIN A DEAL ###########################', '\n')
    
        #On selectionne la colonne pour la date de création 
    date_creation = df.inquiry_created_at
    
    #On selectionne la colonne pour la date de payment 
    date_payment = df.paid_at
    
    #On créer un dataframe
    df_dates = pd.DataFrame({'created_at': date_creation , 'booked_at': date_payment})
    
    # On selectionne les colonnes sans NaN, celle qui ont eu un payment 
    df_dates = df_dates.dropna()
    
    
    #On gère le format date
    df_dates['datetime_booked_at'] = pd.to_datetime(df_dates['booked_at'])
    df_dates.drop(['booked_at'], axis=1, inplace=True)
    df_dates['datetime_created_at'] = pd.to_datetime(df_dates['created_at'])
    df_dates.drop(['created_at'], axis=1, inplace=True)
    
    # On fait la différence en la date de payment et la date de création 
    time_to_win_deals = df_dates['datetime_booked_at'] - df_dates['datetime_created_at']


    # On rajoute au dataframe une derniere colonne : le time to win 
    df['time_to_win'] = time_to_win_deals
    
    # On créer notre target     
    liste_day = list()
    for time in time_to_win_deals:
        liste_day.append(time.days)
    y = liste_day
    
    # On créer notre dataset de prédiction
    inquiry_features = ['paid_at','inquiry_created_at','duration_event_days', 'total_price', 'main_nbr_admin_message', 'main_nbr_renter_message', 'main_nbr_renter_message.1' ]
    X = df[inquiry_features].dropna()
    
    # On enlève les colonnes qui ne servent pas à notre prédicteur
    del X['paid_at']
    del X['inquiry_created_at']
    
    return X,y

def payment_date(df):
    
    # On récupère notre X et y 
    X, y = time_to_win_X_y(df)
    
    # On selectionne les features que l'on veut 
    inquiry_features = ['duration_event_days', 'total_price', 'main_nbr_admin_message', 'main_nbr_renter_message', 'main_nbr_renter_message.1' ]
    
    # On créer notre X_test 
    X_test = df[inquiry_features].dropna()
    
    # On choisit notre classifieur 
    my_model = RandomForestClassifier(n_estimators = 2)
    
    # On l'entraine 
    my_model.fit(X, y)
    
    # On fait nos prédictions 
    predictions_dates = my_model.predict(X_test)
    
    # On gere le format date     
    df['datetime_created_at'] = pd.to_datetime(df['inquiry_created_at'].dropna())
    df.drop(['inquiry_created_at'], axis=1, inplace=True)
    

    # On ajoute à la date de création notre prédiction 
    jours_payment = list()
    for i , date  in zip(df.datetime_created_at , predictions_dates) : 
        jours_payment.append(i + timedelta(int(date)))
    
    # A cause de la derniere ligne 
    #jours_payment.append(0)
    
    return jours_payment 



