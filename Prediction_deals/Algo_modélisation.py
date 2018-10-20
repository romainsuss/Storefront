#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:55:04 2018

@author: romain
"""

import numpy as np
import matplotlib.pyplot as plt
import random 
import csv
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split


### LECTURE DU FICHIER ###
inquiry_data = pd.read_csv('inquiry_prediction_dataset.csv') #On importe le fichier csv 



#On selectionne la Prediction Target
y = inquiry_data.status
#On choisit les features 
inquiry_features = ['duration', 'total_price', 'nbr_messages_by_owner', 'nbr_messages_by_renter', 'nbr_messages_by_admin' ]
#Par convention on appelle les features X 
X = inquiry_data[inquiry_features]

def Algo_gradientboosting(X,y):

    X,y= shuffle(X,y) #On mélange les données 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #On split nos données
    
    
    my_model = GradientBoostingClassifier(n_estimators=20)
    # Add silent=True to avoid printing out updates with each cycle
    my_model.fit(X_train, y_train)
    # make predictions
    predictions = my_model.predict(X_test)
    #On regarde notre precision
    precision = my_model.score(X_test, y_test)

    #print('Précision de notre classifieur sur lensemble de test = ' , precision*100, '%')
    
    return predictions