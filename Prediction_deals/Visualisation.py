#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:09:41 2018

@author: romain
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def visualisation_1_courbe(y,titre):
    
    plt.figure(titre)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    
    plt.plot(months, y)
    plt.xlabel("mois")
    plt.ylabel("Nomdre de deals success")
    plt.title(titre)
    plt.show()
    
    
def visualisation_2_courbe(y,y_pred,titre): 
        
    plt.figure(titre)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    
    # On trace la droite rouge verticale qui annonce le début de la prédiction 
    plt.axvline(x=9,color="r")
    
    plt.plot(months, y, marker='o', label='réalisé')
    plt.plot(months, y_pred, "r:o", label="forecast")
    plt.xlabel("mois")
    plt.ylabel('Nomdre de deals success')
    plt.title('Price_deals_monthly 2018')
    plt.legend()
    
    plt.show()
    
    
    
    