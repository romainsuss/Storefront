#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:53:22 2019

@author: romain
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# Nos modeles 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier


#---------------------------------------------------------------------------#
                    # LECTURE DU FICHIER #
#---------------------------------------------------------------------------#
  
# On lit notre fichier                   
df = pd.read_csv('inquiry.csv' , delimiter=';')
# On gère le format date     
df['datetime_created_at'] = pd.to_datetime(df['inquiry_created_at'])
df.drop(['inquiry_created_at'], axis=1, inplace=True)
# On trie les dates created_at
data_sort = df.sort_values(by=['datetime_created_at'])
# On renome la colonne pour quelle garde le nom initial
inquiry_data = data_sort.rename(columns={"datetime_created_at": "inquiry_created_at"})
# On enlève la dernière ligne
inquiry = inquiry_data.drop([20452])

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

# On trouve le time to win 
time_to_win_deals = X_success['datetime_booked_at'] - X_success['datetime_created_at']
X_success['time_to_win'] = time_to_win_deals.dt.days

# success [901 rows x 20 columns]

 # On créer notre target     
y = X_success['time_to_win']   
 
#X1 = X_success[['nbr_days_before_event']]
        
inquiry_features1 = ['duration_event_days', 'nbr_days_before_event',
   'total_price_norm', 'main_message', 'private_message', 'budget_day']
X1 = X_success[inquiry_features1]




#---------------------------------------------------------------------------#
                    # ON SPLIT NOTRE DATASET #
#---------------------------------------------------------------------------#


# On split notre dataset
X_train, X_test, y_train, y_test = train_test_split(X1, y,
                                                    test_size=0.2)

#---------------------------------------------------------------------------#
                    # ON APPLIQUE NOTRE MODELE #
#---------------------------------------------------------------------------#

model = LinearRegression()

# On l'entraine    
model.fit(X_train,y_train)

#---------------------------------------------------------------------------#
                    # PERF TRAIN #
#---------------------------------------------------------------------------#

        
# On fait nos prédictions
predictions_train = model.predict(X_train)

# On affiche notre performance sur le train 
print('PERFORMANCE SUR LE TRAIN \n')
print('Mean absolute error = ' , mean_absolute_error(y_train, predictions_train), '\n')

#---------------------------------------------------------------------------#
                    # PERF TEST #
#---------------------------------------------------------------------------#


# On fait nos prédictions sur le test
predictions_test = model.predict(X_test)

# On affiche notre performance sur le test 
print('PERFORMANCE SUR LE TEST \n')
print('Mean absolute error = ' , mean_absolute_error(y_test, predictions_test), '\n')
#print(predictions)
#print(y_test)

#---------------------------------------------------------------------------#
                    # PERF DATASET #
#---------------------------------------------------------------------------#

        
# On fait nos prédictions
predictions_dataset = model.predict(X1)

# On affiche notre performance sur le train 
print('PERFORMANCE SUR LE DATASET \n')
print('Mean absolute error = ' , mean_absolute_error(y, predictions_dataset), '\n')


#---------------------------------------------------------------------------#
                    # PERF TEST MONTHLY #
#---------------------------------------------------------------------------#


# On transforme nos predictions en entiers et en date  
pred_int = list()
for i in predictions_dataset:
    pred_int.append(timedelta(int(i)))

# On rajoute la colonne prediction
X_success['time_to_win_pred'] = pred_int
 
# On rajoute une colonne pour le jour de payment prédit 
X_success['jours_payment'] = X_success['datetime_created_at'] + X_success['time_to_win_pred']

#print(X_success[['jours_payment','datetime_created_at','time_to_win_pred']])

# On creer des colonnes avec le numero des mois 
X_success['jours_payment_mois'] = X_success['jours_payment'].dt.month
X_success['booked_at_mois'] = X_success['datetime_booked_at'].dt.month
X_success['created_at_mois'] = X_success['datetime_created_at'].dt.month

# On creer notre colonne predite et la vraie 
X_success['mois_predict'] = X_success['jours_payment_mois'] - X_success['created_at_mois']
X_success['mois'] = X_success['booked_at_mois'] - X_success['created_at_mois'] 

# On remplace les valeurs 2,3,4 par un pour avoir un pb de classification binaire
X_success['mois_predict'].replace(to_replace=[2,3,4],value=1,inplace=True)
X_success['mois'].replace(to_replace=[2,3,4],value=1,inplace=True)

#---------------------------------------------------------------------------#
                    # PERFORMANCE SUR LE DATASET #
#---------------------------------------------------------------------------#

# On affiche notre performance sur le test 
print('-------- PERFORMANCE SUR LE DATASET --------- \n')
print('classification_report  \n',classification_report(X_success['mois_predict'],X_success['mois']))  
print('ROC = ', roc_auc_score(X_success['mois_predict'],X_success['mois']), '\n')

fpr_train, tpr_train, thresholds = metrics.roc_curve(X_success['mois_predict'],X_success['mois'], pos_label=0)



#---------------------------------------------------------------------------#
                    # Courbe ROC #
#---------------------------------------------------------------------------#


# Print ROC curve
plt.figure('Courbe ROC')
plt.plot(tpr_train,fpr_train, color = 'b', label = ('courbe ROC :', round(roc_auc_score(X_success['mois_predict'],X_success['mois']),2)))
plt.plot([0,1],[0,1],color='r',ls="--")
plt.title('Courbe ROC')
plt.legend()
plt.show()


#---------------------------------------------------------------------------#
                    # Coeff #
#---------------------------------------------------------------------------#

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

print('variables X : ', X1.columns, '\n')

print('Coef a : ', model.coef_, '\n')

print('Coef b : ', model.intercept_, '\n')


## Graphique sans matrice
#plt.figure(3) 
#plt.scatter(X1[X1.columns[0]],y, label = 'scatter')
#plt.plot(X1[X1.columns[0]],predictions_dataset,color="red", label = 'reg log' )
#plt.title('regression linéaire')
#plt.ylabel('Time to win')
#plt.xlabel(X1.columns[0])
#
## Graphique avec matrice 
#plt.figure(2) 
#plt.scatter(X1[X1.columns[0]],y, label = 'scatter')
#y = np.dot(model.coef_ , X1.T) + model.intercept_ 
#plt.plot(X1, y,color="red")
#plt.title('regression linéaire')
#plt.ylabel('Time to win')
#plt.xlabel(X1.columns[0])





