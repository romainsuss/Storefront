# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:43:44 2018

@author: aurea
"""

import numpy as np
import matplotlib.pyplot as plt
import random 
import csv
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
data = pd.read_csv('pfe_inquiry_prediction.csv')

features = ['total_price','nbr_messages_by_admin','status']

X = data[features]
X_success = X[X['status'] == 'success']

features2 = ['total_price','nbr_messages_by_admin']
X_success_no_status = X_success[features2]
total_price = X_success['total_price']
nbr_messages = X_success['nbr_messages_by_admin']

model = PCA(n_components=2)
model.fit(X_success_no_status)
reduc = model.transform(X_success_no_status)
print(X_success_no_status[:5])
print(reduc[:5])



plt.scatter(nbr_messages, total_price, c = 'black')
plt.grid()
plt.title('total_price / messages')
#plt.plot(nbr_messages, fitLine, c='r')