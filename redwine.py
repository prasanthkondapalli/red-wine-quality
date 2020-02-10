# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:50:56 2020

@author: Prasanth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('F:/downloads/task')
df.columns
x=df[['fixed acidity','volatile acidity' , 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
y=df[['quality (Target)']]


df.isnull().sum()

#box plot to check outliers
plt.boxplot(x['fixed acidity'])
plt.boxplot(x['volatile acidity'])
plt.boxplot(x['citric acid'])
plt.boxplot(x['residual sugar'])
plt.boxplot(x['chlorides'])
plt.boxplot(x['free sulfur dioxide'])
plt.boxplot(x['total sulfur dioxide'])
plt.boxplot(x['density'])
plt.boxplot(x['pH'])
plt.boxplot(x['sulphates'])
plt.boxplot(x['alcohol'])


#data cleaning
per=x['fixed acidity'].quantile([0.0,0.95]).values
x['fixed acidity']=x['fixed acidity'].clip(per[0],per[1])

per=x['volatile acidity'].quantile([0.0,0.95]).values
x['volatile acidity']=x['volatile acidity'].clip(per[0],per[1])

per=x['citric acid'].quantile([0.0,0.95]).values
x['citric acid']=x['citric acid'].clip(per[0],per[1])

per=x['residual sugar'].quantile([0.0,0.9]).values
x['residual sugar']=x['residual sugar'].clip(per[0],per[1])

per=x['chlorides'].quantile([0.5,0.89]).values
x['chlorides']=x['chlorides'].clip(per[0],per[1])

per=x['free sulfur dioxide'].quantile([0.0,0.9]).values
x['free sulfur dioxide']=x['free sulfur dioxide'].clip(per[0],per[1])

per=x['total sulfur dioxide'].quantile([0,0.89]).values
x['total sulfur dioxide']=x['total sulfur dioxide'].clip(per[0],per[1])

per=x['density'].quantile([0.5,0.9]).values
x['density']=x['density'].clip(per[0],per[1])


per=x['pH'].quantile([0.5,0.9]).values
x['pH']=x['pH'].clip(per[0],per[1])



per=x['sulphates'].quantile([0,0.89]).values
x['sulphates']=x['sulphates'].clip(per[0],per[1])

per=x['alcohol'].quantile([0,0.97]).values
x['alcohol']=x['alcohol'].clip(per[0],per[1])

# train  test  spit

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=80)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)

ypred =reg.predict(x_test)
ypred_tr=reg.predict(x_train)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

accuracy_score(y_test.values,ypred) #69
accuracy_score(y_train.values,ypred_tr) #59
''' logistic regression completed'''


#decission tree 
from sklearn import tree
dec=tree.DecisionTreeClassifier()
dec.fit(x_train,y_train)

ypred_dec =dec.predict(x_test)
ypred_dec_tr=dec.predict(x_train)

accuracy_score(y_test,ypred_dec) #60
accuracy_score(y_train,ypred_dec_tr) #100

#random forest
from sklearn.ensemble import RandomForestClassifier 
ran=RandomForestClassifier()
ran.fit(x_train,y_train)

ypred_ran_ts=ran.predict(x_test)
ypred_ran_tr=ran.predict(x_train)

accuracy_score(y_test.values,ypred_ran_ts)#69
accuracy_score(y_train,ypred_ran_tr)#98
































