# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:27:41 2018

@author: aurea
"""
from __future__ import division

import numpy as np
import pandas as pd
from math import *
from statsmodels import robust
import statistics
from astropy.stats import median_absolute_deviation

inquiry_data = pd.read_csv('inquiry.csv', delimiter=';', low_memory=False)
total_price = inquiry_data['total_price']
duration_event_days = inquiry_data['duration_event_days']
inquiry_data['duration_event_days'] = inquiry_data['duration_event_days'] + 1
nbr_days_before_event = inquiry_data['nbr_days_before_event']

duration_event_days_copy = duration_event_days.copy()
duration = duration_event_days_copy.replace(0.0, 1.0)
inquiry_data['budget_day'] = total_price / duration
budget_day = inquiry_data['budget_day'] 

features = ['inquiry_id','duration_event_days','nbr_days_before_event','budget_day','event_type','status_deal']
X = inquiry_data[features].dropna()
X_head = inquiry_data[features].head(2)

print(X.duration_event_days, '\n')
print(X[X['duration_event_days'] == 1], '\n')
print(X_head.budget_day, '\n')
print(X_head.duration_event_days, '\n')
print(X_head.nbr_days_before_event, '\n')

def gauss_fct(x,m,s):  
    return exp(-(x**2)/(2*s**2))

def distance_metric(X_first, X_second):
    
    budget_weight = gauss_fct((X_first.budget_day-X_second.budget_day),0, median_absolute_deviation(X['budget_day']))
    
    duration_weight = gauss_fct((X_first.duration_event_days-X_second.duration_event_days),0, median_absolute_deviation(X['duration_event_days']))
    
    anticipation_weight = gauss_fct((X_first.nbr_days_before_event-X_second.nbr_days_before_event),0, median_absolute_deviation(X['nbr_days_before_event']))
    
    if (X_first.event_type == X_second.event_type):
        event_type_weight = 1
    else: 
        event_type_weight = 0
        
    print('budget_weight = ', budget_weight, '\n')
    print('duration_weight = ', duration_weight, '\n')
    print('anticipation_weight = ', anticipation_weight, '\n')
    print('event_type_weight = ', event_type_weight, '\n')
    score = (budget_weight + duration_weight + anticipation_weight + event_type_weight)/4
    
    return score

score_list = list()
list_inquiry1 = list()
list_inquiry2 = list()
    
for i in range(0,len(X_head)-1):
    X_first = X_head.iloc[i,:]
    X_second = X_head.iloc[i+1,:]
    score = distance_metric(X_first,X_second)
    score_list.append(score)
    list_inquiry1.append(X_first)
    list_inquiry2.append(X_second)

print('MAD budget :', median_absolute_deviation(X.budget_day),' \n')
print('MAD duration :', median_absolute_deviation(X.duration_event_days),' \n')
print('MAD time anticipation :', median_absolute_deviation(X.nbr_days_before_event),' \n')

df_score = pd.DataFrame({'inquiry 1': list_inquiry1, 'inquiry 2 ': list_inquiry2, 'score': score_list})
print(df_score.score)
