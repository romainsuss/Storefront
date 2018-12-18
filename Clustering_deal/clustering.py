# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:57:58 2018

@author: aurea
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

inquiry_data = pd.read_csv('inquiry.csv', delimiter=';', low_memory=False)
#inquiry_data = data[data['status_deal'] == 'success']
#print(inquiry_data.head())
#print(inquiry_data.columns)

## On isole chaque colonne pour voir à quoi elle correspond
inquiry_id = inquiry_data['inquiry_id']
#print(inquiry_id)

renter_id = inquiry_data['renter_id']
#print(renter_id)

status_deal = inquiry_data['status_deal']
#print(status_deal)

inquiry_created_at = inquiry_data['inquiry_created_at']
#print(inquiry_created_at)

paid_at = inquiry_data['paid_at']
#print(paid_at)

start_date = inquiry_data['start_date']
#print(start_date)

end_date = inquiry_data['end_date']
#print(end_date)

event_type = inquiry_data['event_type']
#print(event_type)

total_price = inquiry_data['total_price']
#print(total_price)

duration_event_days = inquiry_data['duration_event_days']
#print(duration_event_days)

nbr_days_before_event = inquiry_data['nbr_days_before_event']
#print(nbr_days_before_event)

main_nbr_admin_message = inquiry_data['main_nbr_admin_message']
#print(main_nbr_admin_message)

main_nbr_renter_message = inquiry_data['main_nbr_renter_message']
#print(main_nbr_renter_message)

main_nbr_renter_message_1 = inquiry_data['main_nbr_renter_message.1']
#print(main_nbr_renter_message_1)

private_lo_nbr_admin_message = inquiry_data['private_lo_nbr_admin_message']
#print(private_lo_nbr_admin_message)

private_lo_nbr_admin_message_1 = inquiry_data['private_lo_nbr_admin_message.1']
#print(private_lo_nbr_admin_message_1)

private_renter_nbr_admin_message = inquiry_data['private_renter_nbr_admin_message']
#print(private_renter_nbr_admin_message)

private_renter_nbr_admin_message_1 = inquiry_data['private_renter_nbr_admin_message.1']
#print(private_renter_nbr_admin_message_1)

company_size = inquiry_data['company_size']
#print(company_size)

## on rajoute une colonne budget_day
duration_event_days_copy = duration_event_days.copy()
duration = duration_event_days_copy.replace(0.0, 1.0)
inquiry_data['budget_day'] = total_price / duration
budget_day = inquiry_data['budget_day']
#print(budget_day)

## pour connaitre les différents type d'event
#print(event_type.unique())

pop_up_store = inquiry_data[inquiry_data['event_type'] == 'Pop-Up Store']
#print(pop_up_store)
fashion_show = inquiry_data[inquiry_data['event_type'] == 'Fashion Show']
#print(fashion_show)
fashion_showroom = inquiry_data[inquiry_data['event_type'] == 'Fashion Showroom']
#print(fashion_showroom)
private_sale = inquiry_data[inquiry_data['event_type'] == 'Private Sale']
#print(private_sale)
product_launch = inquiry_data[inquiry_data['event_type'] == 'Product Launch']
#print(product_launch)
art_opening = inquiry_data[inquiry_data['event_type'] == 'Art Opening']
#print(art_opening)
corporate_event = inquiry_data[inquiry_data['event_type'] == 'Corporate Event']
#print(corporate_event)
food_event = inquiry_data[inquiry_data['event_type'] == 'Food Event']
#print(food_event)
late_night_event = inquiry_data[inquiry_data['event_type'] == 'Late Night Event (after 10pm)']
#print(late_night_event)
photoshoot = inquiry_data[inquiry_data['event_type'] == 'Photoshoot & Filming']
#print(photoshoot)

## A noter : aucun success sur les shopping_mall
shopping_mall = inquiry_data[inquiry_data['event_type'] == 'Shopping Mall']
#print(shopping_mall)

features = ['duration_event_days','nbr_days_before_event','budget_day','event_type', 'status_deal']
features2 = ['duration_event_days','nbr_days_before_event','budget_day']

typo_deal = inquiry_data[features].dropna()
#print(typo_deal.head(1))
typo_deal_without_event = inquiry_data[features2].dropna()
#print(typo_deal_without_event)
typo_deal_pop_up_store =  pop_up_store[features2]
#print(pop_up_store)
typo_deal_fashion_showroom = fashion_showroom[features2]
#print(fashion_showroom)
typo_deal_private_sale = private_sale[features2]
#print(private_sale)
typo_deal_product_launch = product_launch[features2]
#print(product_launch)
typo_deal_art_opening = art_opening[features2]
#print(art_opening)
typo_deal_corporate_event = corporate_event[features2]
#print(corporate_event)
typo_deal_food_event = food_event[features2]
#print(food_event)
typo_deal_late_night_event = late_night_event[features2]
#print(late_night_event)
typo_deal_photoshoot = photoshoot[features2]
#print(typo_deal_photoshoot)

#values_pop_up_store = typo_deal_pop_up_store.values
#values_fashion_showroom = typo_deal_fashion_showroom.values
#values_private_sale = typo_deal_private_sale.values
#values_product_launch = typo_deal_product_launch.values
#values_art_opening = typo_deal_art_opening.values
#values_corporate_event = typo_deal_corporate_event.values
#values_food_event = typo_deal_food_event.values
#values_late_night_event = typo_deal_late_night_event.values
#values_photoshoot = typo_deal_photoshoot.values

## Clustering 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def number_clusters(coord):
    
    Nc = range(1, 9)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    kmeans
    score = [kmeans[i].fit(coord).score(coord) for i in range(len(kmeans))]
    print(score)
    f = plt.figure(1)
    plt.plot(Nc,score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    f.show()
    j=0
    while(score[j] < -0.8 * 10**9):
        score[j] = score[j+1]
        j = j+1
    nb_clusters = j+1

    return nb_clusters

#number = number_clusters(typo_deal_without_event.values)
X = StandardScaler().fit_transform(typo_deal_without_event.values)
pca = PCA(n_components=2)
x_2d = pca.fit_transform(X)
print('Composants ACP : ', pca.components_)
#p = plt.figure(2)
#plt.scatter(x_2d[:,0],x_2d[:,1], c='goldenrod',alpha=0.5)
#plt.ylim(-10,30)
#p.show()


kmeans = KMeans(n_clusters=3)
y_km = kmeans.fit_predict(x_2d)
typo_deal['labels'] = kmeans.labels_
#g = plt.figure(3)
#plt.scatter(typo_deal_without_event.values[y_km ==0,1], typo_deal_without_event.values[y_km == 0,0], s=10, c='red')
#plt.scatter(typo_deal_without_event.values[y_km ==1,1], typo_deal_without_event.values[y_km == 1,0], s=10, c='yellow')
#plt.scatter(typo_deal_without_event.values[y_km ==2,1], typo_deal_without_event.values[y_km == 2,0], s=10, c='blue')
#plt.loglog()
#g.show()

# Create a temp dataframe from our PCA projection data "x_3d"
df = pd.DataFrame(x_2d)
df['X_cluster'] = y_km
#sns.pairplot(df, hue='X_cluster', palette= 'Dark2', diag_kind='kde',size=2.70)
    
## Etudes statistiques
label0 = typo_deal[typo_deal['labels'] == 0]
#print(label0)
label1 = typo_deal[typo_deal['labels'] == 1]
#print(label1)
label2 = typo_deal[typo_deal['labels'] == 2]
#print(label2)

print()
print('Description du premier cluster :')
print(label0.describe())
print()
print('Description du deuxième cluster :')
print(label1.describe())
print()
print('Description du troisième cluster :')
print(label2.describe())
print()

print('Premier cluster : Nombre de success :', len(label0[label0['status_deal'] == 'success'].index), '(', (len(label0[label0['status_deal'] == 'success'].index) / len(label0.index)) *100, '%)')
print('Premier cluster : Nombre de fail :', len(label0[label0['status_deal'] == 'failed'].index), '(', (len(label0[label0['status_deal'] == 'failed'].index) / len(label0.index)) *100, '%)')
print()
print('Deuxième cluster : Nombre de success :', len(label1[label1['status_deal'] == 'success'].index), '(', (len(label1[label1['status_deal'] == 'success'].index) / len(label1.index)) *100, '%)')
print('Deuxième cluster : Nombre de fail :', len(label1[label1['status_deal'] == 'failed'].index), '(', (len(label1[label1['status_deal'] == 'failed'].index) / len(label1.index)) *100, '%)')
print()
print('Troisième cluster : Nombre de success :', len(label2[label2['status_deal'] == 'success'].index), '(', (len(label2[label2['status_deal'] == 'success'].index) / len(label2.index)) *100, '%)')
print('Troisième cluster : Nombre de fail :', len(label2[label2['status_deal'] == 'failed'].index), '(', (len(label2[label2['status_deal'] == 'failed'].index) / len(label2.index)) *100, '%)')
print()

print(label0.set_index(['duration_event_days','nbr_days_before_event','budget_day','event_type','status_deal']).groupby('event_type').count())
print()
print('Art opening : Nombre de success :', len(art_opening[art_opening['status_deal'] == 'success'].index), '(', (len(art_opening[art_opening['status_deal'] == 'success'].index) / len(art_opening.index)) *100, '%)')
print('Art opening : Nombre de fail :', len(art_opening[art_opening['status_deal'] == 'failed'].index), '(', (len(art_opening[art_opening['status_deal'] == 'failed'].index) / len(art_opening.index)) *100, '%)')
print()
print('Corporate Event : Nombre de success :', len(corporate_event[corporate_event['status_deal'] == 'success'].index), '(', (len(corporate_event[corporate_event['status_deal'] == 'success'].index) / len(corporate_event.index)) *100, '%)')
print('Corporate Event : Nombre de fail :', len(corporate_event[corporate_event['status_deal'] == 'failed'].index), '(', (len(corporate_event[corporate_event['status_deal'] == 'failed'].index) / len(corporate_event.index)) *100, '%)')
print()
print('Fashion Show : Nombre de success :', len(fashion_show[fashion_show['status_deal'] == 'success'].index), '(', (len(fashion_show[fashion_show['status_deal'] == 'success'].index) / len(fashion_show.index)) *100, '%)')
print('Fashion Show : Nombre de fail :', len(fashion_show[fashion_show['status_deal'] == 'failed'].index), '(', (len(fashion_show[fashion_show['status_deal'] == 'failed'].index) / len(fashion_show.index)) *100, '%)')
print()
print('Fashion Showroom : Nombre de success :', len(fashion_showroom[fashion_showroom['status_deal'] == 'success'].index), '(', (len(fashion_showroom[fashion_showroom['status_deal'] == 'success'].index) / len(fashion_showroom.index)) *100, '%)')
print('Fashion Showroom : Nombre de fail :', len(fashion_showroom[fashion_showroom['status_deal'] == 'failed'].index), '(', (len(fashion_showroom[fashion_showroom['status_deal'] == 'failed'].index) / len(fashion_showroom.index)) *100, '%)')
print()
print('Food Event : Nombre de success :', len(food_event[food_event['status_deal'] == 'success'].index), '(', (len(food_event[food_event['status_deal'] == 'success'].index) / len(food_event.index)) *100, '%)')
print('Food Event : Nombre de fail :', len(food_event[food_event['status_deal'] == 'failed'].index), '(', (len(food_event[food_event['status_deal'] == 'failed'].index) / len(food_event.index)) *100, '%)')
print()
print('Late Night Event : Nombre de success :', len(late_night_event[late_night_event['status_deal'] == 'success'].index), '(', (len(late_night_event[late_night_event['status_deal'] == 'success'].index) / len(late_night_event.index)) *100, '%)')
print('Late Night Event : Nombre de fail :', len(late_night_event[late_night_event['status_deal'] == 'failed'].index), '(', (len(late_night_event[late_night_event['status_deal'] == 'failed'].index) / len(late_night_event.index)) *100, '%)')
print()
print('Photoshoot & Filming : Nombre de success :', len(photoshoot[photoshoot['status_deal'] == 'success'].index), '(', (len(photoshoot[photoshoot['status_deal'] == 'success'].index) / len(photoshoot.index)) *100, '%)')
print('Photoshoot & Filming : Nombre de fail :', len(photoshoot[photoshoot['status_deal'] == 'failed'].index), '(', (len(photoshoot[photoshoot['status_deal'] == 'failed'].index) / len(photoshoot.index)) *100, '%)')
print()
print('Pop-up Store : Nombre de success :', len(pop_up_store[pop_up_store['status_deal'] == 'success'].index), '(', (len(pop_up_store[pop_up_store['status_deal'] == 'success'].index) / len(pop_up_store.index)) *100, '%)')
print('Pop-up Store : Nombre de fail :', len(pop_up_store[pop_up_store['status_deal'] == 'failed'].index), '(', (len(pop_up_store[pop_up_store['status_deal'] == 'failed'].index) / len(pop_up_store.index)) *100, '%)')
print()
print('Private Sale : Nombre de success :', len(private_sale[private_sale['status_deal'] == 'success'].index), '(', (len(private_sale[private_sale['status_deal'] == 'success'].index) / len(private_sale.index)) *100, '%)')
print('Private Sale : Nombre de fail :', len(private_sale[private_sale['status_deal'] == 'failed'].index), '(', (len(private_sale[private_sale['status_deal'] == 'failed'].index) / len(private_sale.index)) *100, '%)')
print()
print('Product Launch: Nombre de success :', len(product_launch[product_launch['status_deal'] == 'success'].index), '(', (len(product_launch[product_launch['status_deal'] == 'success'].index) / len(product_launch.index)) *100, '%)')
print('Product Launch : Nombre de fail :', len(product_launch[product_launch['status_deal'] == 'failed'].index), '(', (len(product_launch[product_launch['status_deal'] == 'failed'].index) / len(product_launch.index)) *100, '%)')
print()
print('Shopping Mall : Nombre de success :', len(shopping_mall[shopping_mall['status_deal'] == 'success'].index), '(', (len(shopping_mall[shopping_mall['status_deal'] == 'success'].index) / len(shopping_mall.index)) *100, '%)')
print('Shopping Mall : Nombre de fail :', len(shopping_mall[shopping_mall['status_deal'] == 'failed'].index), '(', (len(shopping_mall[shopping_mall['status_deal'] == 'failed'].index) / len(shopping_mall.index)) *100, '%)')
print()

#prédicteur de clustering a faire