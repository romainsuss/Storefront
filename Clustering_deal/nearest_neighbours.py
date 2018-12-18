# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:20:45 2018

@author: aurea
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

inquiry_data = pd.read_csv('inquiry.csv', delimiter=';', low_memory=False)

total_price = inquiry_data['total_price']
duration_event_days = inquiry_data['duration_event_days']
nbr_days_before_event = inquiry_data['nbr_days_before_event']

duration_event_days_copy = duration_event_days.copy()
duration = duration_event_days_copy.replace(0.0, 1.0)
inquiry_data['budget_day'] = total_price / duration
budget_day = inquiry_data['budget_day']

features = ['duration_event_days','nbr_days_before_event','budget_day','event_type','status_deal']
X = inquiry_data[features].dropna()
#y = inquiry_data['status_deal'].dropna()
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

## paramètre à mettre dans la fonction : 
## inquiry de la forme : (dataframe, list(duration_event_days,nbr_days_before_event,budget_day,event_type)
def percent_similar_inquiry(X, input_inquiry):
    input_train = pd.DataFrame({'duration_event_days': [input_inquiry[0]], 
                                'nbr_days_before_event': [input_inquiry[1]],
                                'budget_day': [input_inquiry[2]],
                                'event_type': [input_inquiry[3]]})
    print('Inquiry à comparer : ')
    print(input_train, '\n')
    
    score_list_duration = list()
    score_list_time_anticipation = list()
    score_list_budget_day = list()
    score_list_event_type = list()
    
    for index, row in X.iterrows():
        list_duration = (abs(row['duration_event_days'] - input_train['duration_event_days']) <= (20/100) * row['duration_event_days']).tolist()
        if (list_duration[0] == True):
            score_list_duration.append(0.25)
        else:
            score_list_duration.append(0)
        
        list_time_anticipation = (abs(row['nbr_days_before_event'] - input_train['nbr_days_before_event']) <= (20/100) * row['nbr_days_before_event']).tolist()
        if (list_time_anticipation[0] == True):
            score_list_time_anticipation.append(0.25)
        else:
            score_list_time_anticipation.append(0)
        
        list_budget_day = (abs(row['budget_day'] - input_train['budget_day']) <= (20/100) * row['budget_day']).tolist()
        if (list_budget_day[0] == True):
            score_list_budget_day.append(0.25)
        else:
            score_list_budget_day.append(0)
    
        list_event_type = (row['event_type'] == input_train['event_type']).tolist()
        if (list_event_type[0] == True):
            score_list_event_type.append(0.25)
        else:
            score_list_event_type.append(0)
    
    score_list = [w + x + y + z for w, x, y, z in zip(score_list_duration, score_list_time_anticipation, score_list_budget_day, score_list_event_type)]
    X['similarity_score'] = score_list
    
    similar_inquiry = X[X['similarity_score'] >= 0.75]
    print('Inquiry similaires : ','\n')
    print(similar_inquiry,'\n')
    print('Pourcentage de success sur les inquiry similaires : ', (len(similar_inquiry[similar_inquiry['status_deal'] == 'success'].index) / len(similar_inquiry.index)) * 100, ' %','\n')
    print('Pourcentage de fail sur les inquiry similaires : ', (len(similar_inquiry[similar_inquiry['status_deal'] == 'failed'].index) / len(similar_inquiry.index)) * 100, ' %','\n')
    
    return 

inquiry_to_test = [3,52,1000,'Fashion Showroom']
inquiry_to_test_2 = [46,166,2090,'Product Launch']
inquiry_to_test_3 = [3,82,51550,'Art Opening']
percent_similar_inquiry(X, inquiry_to_test)
percent_similar_inquiry(X, inquiry_to_test_2)
percent_similar_inquiry(X, inquiry_to_test_3)