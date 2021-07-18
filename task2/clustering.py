# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 12:12:53 2021

@author: Surekha Padmam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("C:/Users/Surekha Padmam/Desktop/ML Datasets/DS,BA internship/Iris.csv")
print(df.head())
print(df.tail())

x=df.iloc[:,1:5]
y=df["Species"]
print(x.shape)
print(x.head())
print("___________________")
print(y.shape)
print(x.head())

kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)

print(kmeans5.cluster_centers_)
#Elbow method
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
    
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.scatter([3],[80])
plt.show()

# from the plot, we can see that the elbow formed between 2 and 4 which is 3.
#therefore k=3.

kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)
print(kmeans3.cluster_centers_)
plt.scatter(x.iloc[y_kmeans3==0,0], x.iloc[y_kmeans3==0,1],s = 100, c = 'red', label = 'Setosa')
plt.scatter(x.iloc[y_kmeans3==1,0], x.iloc[y_kmeans3==1,1],s = 100, c = 'blue', label = 'Versicolour')
plt.scatter(x.iloc[y_kmeans3==2,0], x.iloc[y_kmeans3==2,1],s=100,c="green",label="Verginica")
plt.legend()
plt.show()