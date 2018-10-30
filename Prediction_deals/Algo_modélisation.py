#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:55:04 2018

@author: romain
"""


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier



### ALGORITHME DE CLASSIFICATION ###


def Algo_gradientboosting_classification(X,y):
    
    print('################## ALGORITHME DE PREDICTIONS ###################')
    
    # On découpe notre dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) #On split nos données
    
    # On choisit un classifieur 
    my_model = GradientBoostingClassifier(n_estimators=20)
    
    # On l'entraine 
    my_model.fit(X, y)
    
    # On fait nos prédictions 
    predictions = my_model.predict(X_test)
    
    #On regarde notre precision
    precision = my_model.score(X_test, y_test)
    
    # On affiche notre performance 
    print(classification_report(y_test, predictions))
    print('Précision de notre classifieur sur lensemble de test = ' , precision*100, '%', '\n')
    
    return predictions


def Algo_randomforest_classification(X,y):
    
    print('################## ALGORITHME DE PREDICTIONS SUCCESS/FAILED ###################')
    
    # On découpe notre dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) #On split nos données
    
    # On choisit un classifieur
    my_model = RandomForestClassifier(n_estimators = 2)
    
    # On l'entraine    
    my_model.fit(X, y)
    
    # On fait nos prédictions
    predictions = my_model.predict(X_test)
    
    #On regarde notre precision
    precision = my_model.score(X_test, y_test)
    
    # On affiche notre performance
    print(classification_report(y_test, predictions))
    print('Précision de notre classifieur sur lensemble de test = ' , precision*100, '%', '\n')
    
    return predictions



### ALGORITHME DE CLASSIFICATION ###
    



def Algo_randomforest_regression(X,y):
    
    print('################## ALGORITHME DE PREDICTIONS TIME TO WIN ###################')
          
    # On découpe notre dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) #On split nos données
    
    # On choisit un classifieur
    my_model = RandomForestClassifier(n_estimators = 2)
    
    # On l'entraine   
    my_model.fit(X, y)
    
    # On fait nos prédictions
    predictions = my_model.predict(X_test)

    # On affiche notre performance
    from sklearn.metrics import mean_absolute_error
    print('Mean absolute error = ' , mean_absolute_error(y_test, predictions), '\n')
    
    return predictions


  
def Algo_gradientboosting_regression(X,y):
    
    print('################## ALGORITHME DE PREDICTIONS ###################')
    
    # On découpe notre dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0) #On split nos données
    
    # On choisit un classifieur
    my_model = GradientBoostingClassifier(n_estimators=20)
    
    # On l'entraine   
    my_model.fit(X, y)
    
    # On fait nos prédictions
    predictions = my_model.predict(X_test)

    # On affiche notre performance
    from sklearn.metrics import mean_absolute_error
    print('Mean absolute error = ' , mean_absolute_error(y_test, predictions), '\n')
    
    return predictions








