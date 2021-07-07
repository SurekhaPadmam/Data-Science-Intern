# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:25:11 2021

@author: Surekha Padmam
"""
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dt=pd.read_csv("C:/Users/Surekha Padmam/Desktop/ML Datasets/student-scores.csv")
print(dt.head())
xcol=dt.iloc[:,:-1]
ycol=dt.iloc[:,1]

x_train,x_test,y_train,y_test=train_test_split(xcol,ycol,test_size=1/3)


model=LinearRegression()
model.fit(x_train,y_train)
y_train_pred=model.predict(x_train)


plt.scatter(x_train,y_train,marker='+')
plt.plot(x_train,y_train_pred,label='Prediction Line')
plt.title("Training model")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.legend()
plt.show()

y_test_pred=model.predict(x_test)
plt2=plt
plt2.scatter(x_test,y_test,)
plt2.plot(x_test,y_test_pred,label='prediction line')
plt2.title('Testing model')
plt2.xlabel("Study Hours")
plt2.ylabel("Scores")
plt2.legend()

sol=model.predict([[9.25]])
print('A student who studies 9.25 hrs/day will get',sol,'Scores according to our prediction model')

plt2.scatter([9.25],[sol],marker='*')
plt2.show()