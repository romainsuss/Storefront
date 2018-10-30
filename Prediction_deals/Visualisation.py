#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:09:41 2018

@author: romain
"""

import numpy as np
import matplotlib.pyplot as plt


def visualisation_1_courbe(y,titre):
    
    x = np.arange(1,13) # Pour les mois , possibilité de mettre nom des mois 
    plt.plot(x, y)
    plt.xlabel("mois")
    plt.ylabel("total price")
    plt.title(titre)
    
    plt.show()
    
    
def visualisation_2_courbe(y,y_pred): 
    
    print('############### COMPARAISON REALISE VS FORECAST ###############' )
    
    x = np.arange(1,13)  # Pour les mois , possibilité de mettre nom des mois
    
    # On trace la droite rouge verticale qui annonce le début de la prédiction 
    plt.axvline(x=6,color="r")
    
    plt.plot(x, y, marker='o', label='réalisé')
    plt.plot(x, y_pred, "r:o", label="forecast")
    plt.xlabel("mois")
    plt.ylabel("total price")
    plt.title('Price_deals_monthly 2018')
    plt.legend()
    
    plt.show()
    
    
    
    