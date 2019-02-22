#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:46:26 2019

@author: romain
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE  #pip install imblearn    
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)

       

from sklearn import preprocessing
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

# Nos modeles 
from sklearn.tree import DecisionTreeClassifier


# Random forest 
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
#https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

# Unbalanced dataset 
#https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
#https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set
#https://github.com/IBM/xgboost-financial-predictions/blob/master/notebooks/predict_bank_cd_subs_by_xgboost_clf_for_imbalance_dataset.ipynb

# SMOTE 
#https://www.kaggle.com/chtaret/fraud-detection-with-smote-and-randomforest

#Sampling methods
#https://medium.com/anomaly-detection-with-python-and-r/sampling-techniques-for-extremely-imbalanced-data-part-i-under-sampling-a8dbc3d8d6d8

# precision - recall 
#https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
#---------------------------------------------------------------------------#
                    # LECTURE DU FICHIER #
#---------------------------------------------------------------------------#
                    
df = pd.read_csv('Inquiry.csv' , delimiter=',')
# On gere le format date     
df['datetime_created_at'] = pd.to_datetime(df['inquiry_created_at'])
df.drop(['inquiry_created_at'], axis=1, inplace=True)
# On trie les dates created_at
data_sort = df.sort_values(by=['datetime_created_at'])
# On renome la colonne pour quelle garde le nom initial
inquiry_data = data_sort.rename(columns={"datetime_created_at": "inquiry_created_at"})


#---------------------------------------------------------------------------#
                    # PROCESSING #
#---------------------------------------------------------------------------#


# On selectionne la prediction target
y = inquiry_data.status_deal.dropna()
y.replace(to_replace=['failed'],value=0,inplace=True)
y.replace(to_replace=['success'],value=1,inplace=True)

#On choisit les features 
inquiry_features1 = ['currency','event_type', 'main_nbr_admin_message', 'total_price', 'main_nbr_renter_message', 'private_renter_nbr_admin_message', 'nbr_days_before_event', 'duration_event_days', 'private_lo_nbr_admin_message', 'private_lo_nbr_lo_message','private_renter_nbr_renter_message']
X1 = inquiry_data[inquiry_features1].dropna()

# On normalise nos currency en euro 
X_norm = X1[X1['currency'] == 'USD'].total_price * 0.87
X_norm=X_norm.append(X1[X1['currency'] == 'GBP'].total_price * 1.10)
X_norm=X_norm.append(X1[X1['currency'] == 'HKD'].total_price * 0.11)
X_norm=X_norm.append(X1[X1['currency'] == 'SGD'].total_price * 0.63)
X_norm=X_norm.append(X1[X1['currency'] == 'CAD'].total_price * 0.64)
X_norm=X_norm.append(X1[X1['currency'] == 'AUD'].total_price * 0.62)
X_norm=X_norm.append(X1[X1['currency'] == 'CZK'].total_price * 0.038)
X_norm=X_norm.append(X1[X1['currency'] == 'EUR'].total_price)

# On rajoute une colonne avec tous les prix en euro 
X1['total_price_norm'] = X_norm
# On enleve la colonne currency 
del X1['currency']

# On rassemble les colonnes messages
X1['main_message'] = X1['main_nbr_admin_message'] + X1['main_nbr_renter_message']
X1['private_message'] = X1['private_lo_nbr_admin_message'] + X1['private_lo_nbr_lo_message'] + X1['private_renter_nbr_admin_message'] + X1['private_renter_nbr_renter_message']

# On calcule le budget day 
total_price = X1['total_price_norm']
duration_event_days = X1['duration_event_days']
duration_event_days_copy = duration_event_days.copy()
duration = duration_event_days_copy.replace(0.0, 1.0)
X1['budget_day'] = total_price / duration

# On encode nos données
X2 = pd.get_dummies(X1)

inquiry_features2 = ['nbr_days_before_event','duration_event_days', 
       'total_price_norm', 'main_message', 'private_message', 'budget_day',
       'event_type_Art Opening', 'event_type_Corporate Event',
       'event_type_Fashion Show', 'event_type_Fashion Showroom',
       'event_type_Food Event', 'event_type_Late Night Event (after 10pm)',
       'event_type_Photoshoot & Filming', 'event_type_Pop-Up Store',
       'event_type_Private Sale', 'event_type_Product Launch',
       'event_type_Shopping Mall']

X_final = X2[inquiry_features2]


#---------------------------------------------------------------------------#
                    # ON SPLIT NOTRE DATASET #
#---------------------------------------------------------------------------#



# On split notre dataset
X_train, X_test, y_train, y_test = train_test_split(X_final, y,
                                                    stratify=y, 
                                                    test_size=0.2)



#---------------------------------------------------------------------------#
                    # ON APPLIQUE NOTRE MODELE #
#---------------------------------------------------------------------------#


model = DecisionTreeClassifier(max_depth=6)


    
#---------------------------------------------------------------------------#
                    # PREDICTEUR #
#---------------------------------------------------------------------------#

# On choisit un classifieur
my_model = model

# Look at parameters used by our current forest
#print('Parameters currently in use:\n')
#print(my_model.get_params(),"\n" )


# On l'entraine    
my_model.fit(X_train,y_train)
      
#---------------------------------------------------------------------------#
                    # PERF DATASET #
#---------------------------------------------------------------------------#
        
# On fait nos prédictions
predictions = my_model.predict(X_final)

# On affiche notre performance sur le train 
print('PERFORMANCE SUR LE DATASET \n')
print(classification_report(y, predictions))
print('ROC = ', roc_auc_score(y, predictions), '\n')

    #On affiche l'average précision 
average_precision = average_precision_score(y, predictions)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))

#---------------------------------------------------------------------------#
                    # PERF TRAIN #
#---------------------------------------------------------------------------#
        
# On fait nos prédictions
predictions1 = my_model.predict(X_train)

# On affiche notre performance sur le train 
print('PERFORMANCE SUR LE TRAIN \n')
print(classification_report(y_train, predictions1))
print('ROC = ', roc_auc_score(y_train, predictions1), '\n')

    #On affiche l'average précision 
average_precision = average_precision_score(y_train, predictions1)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# calculate precision-recall curve

#plt.figure('precision-recall curve TEST/TRAIN')
#plt.xlabel("Recall")
#plt.ylabel("Précision")
#plt.title('precision-recall curve TEST/TRAIN')
#
#precision, recall, thresholds = precision_recall_curve(y_train, predictions1)
#plt.plot([0, 1], [0.5, 0.5], linestyle='--')
#plt.plot(recall, precision, marker='.')

    
    
#---------------------------------------------------------------------------#
                    # PERF TEST #
#---------------------------------------------------------------------------#


# On fait nos prédictions sur le test
predictions2 = my_model.predict(X_test)

# On affiche notre performance sur le test 
print('PERFORMANCE SUR LE TEST \n')
print(classification_report(y_test, predictions2))   
print('ROC = ', roc_auc_score(y_test, predictions2), '\n')


#On affiche l'average précision 
average_precision = average_precision_score(y_test, predictions2)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# calculate precision-recall curve
#precision, recall, thresholds = precision_recall_curve(y_test, predictions2) 
#plt.plot(recall, precision, marker='.')


#---------------------------------------------------------------------------#
                    # VISUALISATION #
#---------------------------------------------------------------------------#


from sklearn.externals.six import StringIO  
from IPython.display import Image  , display
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(my_model, out_file=dot_data,  feature_names = inquiry_features2,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
display(Image(graph.create_png()))


#---------------------------------------------------------------------------#
                    # Features importances #
#---------------------------------------------------------------------------#
                  

features = ['nbr_days_before_event','duration_event_days', 
       'total_price_norm', 'main_message', 'private_message', 'budget_day',
       'event_type_Art Opening', 'event_type_Corporate Event',
       'event_type_Fashion Show', 'event_type_Fashion Showroom',
       'event_type_Food Event', 'event_type_Late Night Event (after 10pm)',
       'event_type_Photoshoot & Filming', 'event_type_Pop-Up Store',
       'event_type_Private Sale', 'event_type_Product Launch',
       'event_type_Shopping Mall']
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure('features importances')
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
                    
                    
#---------------------------------------------------------------------------#
                    # max_depth #
#---------------------------------------------------------------------------#


max_depths = [1, 2, 3,4,5,6,7,8,9,10]
train_results1 = []
test_results1 = []
train_results2 = []
test_results2 = []
train_results3 = []
test_results3 = []

for max_depth in max_depths:
   rf = DecisionTreeClassifier(max_depth=max_depth)
   rf.fit(X_train, y_train)
   
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   average_precision1 = average_precision_score(y_train, train_pred)
   precision1, recall1, thresholds = precision_recall_curve(y_train, train_pred)
   #train_results1.append(average_precision1)
   train_results2.append(precision1[1])
   train_results3.append(recall1[1])


   
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   average_precision2 = average_precision_score(y_test, y_pred)
   precision2, recall2, thresholds = precision_recall_curve(y_test, y_pred)
   #test_results1.append(average_precision2)
   test_results2.append(precision2[1])
   test_results3.append(recall2[1])

plt.figure('max depth')

#line1, = plt.plot(max_depths, train_results1, "b", label="Train average")
#line2,= plt.plot(max_depths, test_results1, "b--", label="Test average")

line3, = plt.plot(max_depths, train_results2, "r", label="Train précision")
line4,= plt.plot(max_depths, test_results2, "r--", label="Test précision")

line5, = plt.plot(max_depths, train_results3, "g", label="Train recall")
line6,= plt.plot(max_depths, test_results3, "g--", label="Test recall")


plt.figure('max depth')
plt.ylabel("Score")
plt.xlabel("max_depth")


#---------------------------------------------------------------------------#
                    # Scoring #
#---------------------------------------------------------------------------#

# Probabilité d'avoir 0 ; Donc de close le même mois
pred_score = my_model.predict_proba(X_final)[:,1]

# On rajoute la colonne des scores 
X_final['pred_score'] = pred_score

#print(X_success)

# On trie les dates created_at
data_score = X_final.sort_values(by=['pred_score'] , ascending=False)

data_score['Y'] = y

comparaison = data_score[['pred_score','Y']]



