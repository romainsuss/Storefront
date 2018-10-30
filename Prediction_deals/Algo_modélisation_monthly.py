#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:00:54 2018

@author: romain
"""


import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier



### LECTURE DU FICHIER ###
inquiry_data = pd.read_csv('inquiry.csv' , delimiter=';') #On importe le fichier csv 

#On choisit les features 
inquiry_features = ['status_deal','paid_at','duration_event_days', 'total_price', 'main_nbr_admin_message', 'main_nbr_renter_message', 'main_nbr_renter_message.1' ]
inquiry_features_X = ['duration_event_days', 'total_price', 'main_nbr_admin_message', 'main_nbr_renter_message', 'main_nbr_renter_message.1' ]

df = inquiry_data[inquiry_features]

# On gere le format date
df['datetime'] = pd.to_datetime(df['paid_at'])
df = df.set_index('datetime')
df.drop(['paid_at'], axis=1, inplace=True)

####################################################################

#On créer notre X et y pour la prédiction
X = inquiry_data[inquiry_features_X].dropna()
y = inquiry_data.status_deal.dropna()


### ALGORITHME ###

def Algo_gradientboosting(X,y,X_month):
    
    print('################## ALGORITHME DE PREDICTIONS ###################')

    # On découpe notre dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #On split nos données
    
    # On choisit un classifieur
    my_model = GradientBoostingClassifier(n_estimators=20)
    
    # On l'entraine    
    my_model.fit(X, y)
    
    # On fait nos prédictions
    predictions = my_model.predict(X_month)
    
    #On regarde notre precision
    precision = my_model.score(X_test, y_test)

    # On affiche notre performance
    print(classification_report(y_test, predictions))
    print('Précision de notre classifieur sur lensemble de test = ' , precision*100, '%', '\n')
    
    return predictions



def Algo_randomforest_monthly(X,y,X_month,y_month,mois):
    
    print('################## ALGORITHME DE PREDICTIONS ###################', '\n')
    print('---- MOIS :', mois, '-----------', '\n')     

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #On split nos données 
    
    # On choisit un classifieur
    my_model = RandomForestClassifier(n_estimators = 2)
    
    # On l'entraine
    my_model.fit(X, y)
    
    # On fait nos prédictions
    predictions = my_model.predict(X_month)
    
    #On regarde notre precision
    precision = my_model.score(X_month, y_month)

    # On affiche notre performance
    print(classification_report(y_month, predictions))
    print('Performance  = ' , precision*100, '%', '\n')
    
    return predictions


def X_y_month(df_month_2018, features):
    
    # On recupere X et y par mois
    X_month_2018 = df_month_2018[features]
    y_month_2018 = df_month_2018.status_deal
    
    # On remet des nouveaux index et on supprime datetime
    X_month_2018 = X_month_2018.reset_index()
    del X_month_2018['datetime']
    y_month_2018 = y_month_2018.reset_index()
    del y_month_2018['datetime']
    
    return X_month_2018,y_month_2018


####################################################################
 
def Monthly_performance():
    
    # Faire des fonctions .... 
    
    # On recupere les données par mois 
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
    
    
    # On recupere le X et y par mois 
    X_janvier_2018 , y_janvier_2018 = X_y_month(df_janvier_2018, inquiry_features_X)
    X_fevrier_2018 , y_fevrier_2018 = X_y_month(df_fevrier_2018, inquiry_features_X)
    X_mars_2018 , y_mars_2018 = X_y_month(df_mars_2018, inquiry_features_X)
    X_avril_2018 , y_avril_2018 = X_y_month(df_avril_2018, inquiry_features_X)
    X_mai_2018 , y_mai_2018 = X_y_month(df_mai_2018, inquiry_features_X)
    X_juin_2018 , y_juin_2018 = X_y_month(df_juin_2018, inquiry_features_X)
    X_juillet_2018 , y_juillet_2018 = X_y_month(df_juillet_2018, inquiry_features_X)
    X_aout_2018 , y_aout_2018 = X_y_month(df_aout_2018, inquiry_features_X)
    X_septembre_2018 , y_septembre_2018 = X_y_month(df_septembre_2018, inquiry_features_X)
    X_octobre_2018 , y_octobre_2018 = X_y_month(df_octobre_2018, inquiry_features_X)
    X_novembre_2018 , y_novembre_2018 = X_y_month(df_novembre_2018, inquiry_features_X)
    X_decembre_2018 , y_decembre_2018 = X_y_month(df_decembre_2018, inquiry_features_X)
    
    
    # On fait nos prédictions par mois 
    Algo_randomforest_monthly(X,y,X_janvier_2018,y_janvier_2018, 'JANVIER')
    Algo_randomforest_monthly(X,y,X_fevrier_2018,y_fevrier_2018, 'FEVRIER')
    Algo_randomforest_monthly(X,y,X_mars_2018,y_mars_2018, 'MARS')
    Algo_randomforest_monthly(X,y,X_avril_2018 , y_avril_2018, 'AVRIL')
    Algo_randomforest_monthly(X,y,X_mai_2018 , y_mai_2018,'MAI')
    Algo_randomforest_monthly(X,y,X_juin_2018 , y_juin_2018, 'JUIN')
    Algo_randomforest_monthly(X,y,X_juillet_2018 , y_juillet_2018, 'JUILLET')
    Algo_randomforest_monthly(X,y,X_aout_2018 , y_aout_2018, 'AOUT')
    Algo_randomforest_monthly(X,y,X_septembre_2018 , y_septembre_2018, 'SEPTEMBRE')
    Algo_randomforest_monthly(X,y,X_octobre_2018 , y_octobre_2018, 'OCTOBRE' )
    #Algo_randomforest_monthly(X,y,X_novembre_2018 , y_novembre_2018, 'NOVEMBRE')
    #Algo_randomforest_monthly(X,y,X_decembre_2018 , y_decembre_2018, 'DECEMBRE')


    return 


    