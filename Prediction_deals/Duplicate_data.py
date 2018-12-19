#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:59:08 2018

@author: romain
"""
import pandas as pd 


df = pd.read_csv('inquiry.csv' , delimiter=';')

X_success = df[df['status_deal'] == 'success']

print(df.columns)

print(len(df['inquiry_id']))

print(len(set(df['inquiry_id'])))

print(len(df['inquiry_id'].unique()))