#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:46:45 2018

@author: romain
"""

from sklearn.ensemble import RandomForestClassifier

def Predictions_success_failed(df):
     
    
    # On créer notre target     
    y = df['status_deal']    
    inquiry_features1 = ['main_nbr_admin_message', 'total_price', 'main_nbr_renter_message', 'private_renter_nbr_admin_message', 'nbr_days_before_event',  'duration_event_days', 'private_lo_nbr_admin_message']
    X = df[inquiry_features1].dropna()
    
    # On découpe notre dataset
    split = int(len(df)*0.8)
    X_train = X[:split]
    #X_test = X[split:]
    y_train = y[:split]
    #y_test = y[split:]


    # On fait notre prédiction
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    
    model.fit(X_train,y_train)
    pred = model.predict(X)
    
    return pred