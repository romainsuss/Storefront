#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:36:09 2018

@author: romain
"""


from sklearn.linear_model import LinearRegression

def time_to_win_predictions(df):
    
    total_price = df['total_price']
    duration_event_days = df['duration_event_days']
    
    duration_event_days_copy = duration_event_days.copy()
    duration = duration_event_days_copy.replace(0.0, 1.0)
    df['budget_day'] = total_price / duration
    
    # On créer notre target     
    y = df['time_to_win']    
    X = df[['budget_day']]
    
    # On découpe notre dataset
    split = int(len(df)*0.8)
    X_train = X[:split]
    
    #print(df['datetime_created_at'][split:])
    #X_test = X[split:]
    y_train = y[:split]
    #y_test = y[split:]


    # On fait notre prédiction
    model = LinearRegression()
    model.fit(X_train,y_train)
    pred = model.predict(X)
    
    return pred 


def time_to_win_pred_forecast(X_success,X_success_pred):
    
     # On créer notre target     
    y = X_success['time_to_win']    
    X = X_success[['nbr_days_before_event']]
    X_pred = X_success_pred[['nbr_days_before_event']]
    
    # On découpe notre dataset
    split = int(len(X)*0.8)
    X_train = X[:split]
    #X_test = X[split:]
    y_train = y[:split]
    #y_test = y[split:]


    # On fait notre prédiction
    model = LinearRegression()
    model.fit(X_train,y_train)
       
    pred = model.predict(X_pred)
    
    return pred 
