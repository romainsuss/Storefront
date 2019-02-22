#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:48:37 2019

@author: romain
"""

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import metrics





#---------------------------------------------------------------------------#
                    # LECTURE DU FICHIER #
#---------------------------------------------------------------------------#
  
# On lit notre fichier                   
df = pd.read_csv('inquiry.csv' , delimiter=',')
# On gère le format date     
df['datetime_created_at'] = pd.to_datetime(df['inquiry_created_at'])
df.drop(['inquiry_created_at'], axis=1, inplace=True)
# On trie les dates created_at
data_sort = df.sort_values(by=['datetime_created_at'])
# On renome la colonne pour quelle garde le nom initial
inquiry = data_sort.rename(columns={"datetime_created_at": "inquiry_created_at"})


#inquiry data [20452 rows x 19 columns]

#---------------------------------------------------------------------------#
                    # DATA PRE PROCESSING #
#---------------------------------------------------------------------------#
 
# On normalise nos currency en euro 
X_norm = inquiry[inquiry['currency'] == 'USD'].total_price * 0.87
X_norm=X_norm.append(inquiry[inquiry['currency'] == 'GBP'].total_price * 1.10)
X_norm=X_norm.append(inquiry[inquiry['currency'] == 'HKD'].total_price * 0.11)
X_norm=X_norm.append(inquiry[inquiry['currency'] == 'SGD'].total_price * 0.63)
X_norm=X_norm.append(inquiry[inquiry['currency'] == 'CAD'].total_price * 0.64)
X_norm=X_norm.append(inquiry[inquiry['currency'] == 'AUD'].total_price * 0.62)
X_norm=X_norm.append(inquiry[inquiry['currency'] == 'CZK'].total_price * 0.038)
X_norm=X_norm.append(inquiry[inquiry['currency'] == 'EUR'].total_price)

# On rajoute une colonne avec tous les prix en euro 
inquiry['total_price_norm'] = X_norm
# On enleve la colonne currency 
del inquiry['currency']

# On rassemble les colonnes messages
inquiry['main_message'] = inquiry['main_nbr_admin_message'] + inquiry['main_nbr_renter_message']
inquiry['private_message'] = inquiry['private_lo_nbr_admin_message'] + inquiry['private_lo_nbr_lo_message'] + inquiry['private_renter_nbr_admin_message'] + inquiry['private_renter_nbr_renter_message']

# On calcule le budget day 
total_price = inquiry['total_price_norm']
duration_event_days = inquiry['duration_event_days']
duration_event_days_copy = duration_event_days.copy()
duration = duration_event_days_copy.replace(0.0, 1.0)
inquiry['budget_day'] = total_price / duration


#---------------------------------------------------------------------------#
                    # DATASET SUCCESS #
#---------------------------------------------------------------------------#
 
# On selectionne les success 
X_success = inquiry[inquiry['status_deal'] == 'success']
#On gère le format date
X_success['datetime_booked_at'] = pd.to_datetime(X_success['paid_at'])
X_success.drop(['paid_at'], axis=1, inplace=True)
X_success['datetime_created_at'] = pd.to_datetime(X_success['inquiry_created_at'])
X_success.drop(['inquiry_created_at'], axis=1, inplace=True)



# On crée une colonne pour le temps que l'on met pour gagner en mois
time_to_win_deals = X_success['datetime_booked_at'].dt.month - X_success['datetime_created_at'].dt.month
X_success['time_to_win_month'] = time_to_win_deals

# On explore notre dataset 
df_same_month = X_success[X_success['time_to_win_month'] == 0]
# 756
df_different_month = X_success[X_success['time_to_win_month'] == 2]
# 330 (292 : 1 et 34 : 2 et 3 : 3 et 4 : 1)

# On remplace les valeurs 2,3,4 par un pour avoir un pb de classification binaire
X_success['time_to_win_month'].replace(to_replace=[2,3,4],value=1,inplace=True)



#---------------------------------------------------------------------------#
                    # MODELISATION #
#---------------------------------------------------------------------------#


inquiry_features1 = ['event_type','duration_event_days', 'nbr_days_before_event',
   'total_price_norm', 'main_message', 'private_message', 'budget_day']
X1 = X_success[inquiry_features1]

# On encode nos données
X2 = pd.get_dummies(X1)
y = X_success['time_to_win_month']


# On split notre dataset 
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)
# On modélise 
model = XGBClassifier(max_depth=4,n_estimators = 100)
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
pred = model.predict(X2)


#---------------------------------------------------------------------------#
                    # PERFORMANCE SUR LE TRAIN #
#---------------------------------------------------------------------------#

# On affiche notre performance sur le test 
print('-------- PERFORMANCE SUR LE TRAIN --------- \n')
print('classification_report train  \n',classification_report(pred_train, y_train))  
print('ROC = ', roc_auc_score(pred_train, y_train), '\n')

fpr_train, tpr_train, thresholds = metrics.roc_curve(pred_train, y_train, pos_label=0)
#---------------------------------------------------------------------------#
                    # PERFORMANCE SUR LE TEST #
#---------------------------------------------------------------------------#

# On affiche notre performance sur le test 
print('-------- PERFORMANCE SUR LE TEST --------- \n')
print('classification_report test  \n',classification_report(pred_test, y_test))  
print('ROC = ', roc_auc_score(pred_test, y_test), '\n')

fpr_train, tpr_train, thresholds = metrics.roc_curve(pred_test, y_test, pos_label=0)

#---------------------------------------------------------------------------#
                    # PERFORMANCE SUR LE DATASET #
#---------------------------------------------------------------------------#

# On affiche notre performance sur le test 
print('-------- PERFORMANCE SUR LE DATASET --------- \n')
print('classification_report dataset  \n',classification_report(pred, y))  
print('ROC = ', roc_auc_score(y, pred), '\n')

fpr_train, tpr_train, thresholds = metrics.roc_curve(y, pred, pos_label=0)


#---------------------------------------------------------------------------#
                    # Courbe ROC #
#---------------------------------------------------------------------------#

plt.figure('Courbe roc')
plt.plot(tpr_train,fpr_train, color = 'b', label = ('Courbe ROC :', round(roc_auc_score(y, pred),2)))
plt.plot([0,1],[0,1],color='r',ls="--")
plt.title('Courbe ROC')
plt.legend()
plt.show() 


#---------------------------------------------------------------------------#
                    # TREE #
#---------------------------------------------------------------------------#

xgb.plot_importance(model)

## plot single tree
#plot_tree(model,num_trees=19)
