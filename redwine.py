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
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#logistic regression
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)

ypred =reg.predict(x_test)
ypred_tr=reg.predict(x_train)

reg.coef_
reg.score


accuracy_score(y_test.values,ypred) #60
accuracy_score(y_train.values,ypred_tr) #61

from sklearn.metrics import r2_score
r2_2=r2_score(y_test,ypred)#26

from sklearn.metrics import classification_report

print(classification_report(y_test.values,ypred))


''' logistic regression completed'''


#decission tree 
from sklearn import tree
dec=tree.DecisionTreeClassifier()
dec.fit(x_train,y_train)

ypred_dec =dec.predict(x_test)
ypred_dec_tr=dec.predict(x_train)

accuracy_score(y_test,ypred_dec) #61
accuracy_score(y_train,ypred_dec_tr) #100

dec.coef_


#random forest
from sklearn.ensemble import RandomForestClassifier 
ran=RandomForestClassifier()
ran.fit(x_train,y_train)

ypred_ran_ts=ran.predict(x_test)
ypred_ran_tr=ran.predict(x_train)

accuracy_score(y_test.values,ypred_ran_ts)#66
accuracy_score(y_train.values,ypred_ran_tr)#98

ran.coef_

#SVM 
from sklearn import svm
supv=svm.SVC()
supv.fit(x_train,y_train)

ypred_svm_ts=supv.predict(x_test)
ypred_svm_tr=supv.predict(x_train)

accuracy_score(y_test.values,ypred_svm_ts)#63
print(accuracy_score(y_train.values,ypred_svm_tr))#67
 

print(confusion_matrix(y_test.values,ypred_svm_ts))
confusion_matrix(y_train.values,ypred_svm_tr)
	
from sklearn.metrics import classification_report

print(classification_report(y_test.values,ypred_svm_ts))


#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

ypred_knn_ts=knn.predict(x_test)
ypred_knn_tr=knn.predict(x_train)

accuracy_score(y_test,ypred_knn_ts)#58
accuracy_score(y_train,ypred_knn_tr)#69
knn.classes_
knn.score

#naive bayes
#gaussian
#from mixed_naive_bayes import MixedNB
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import CategoricalNB
gaus=GaussianNB()
gaus.fit(x_train,y_train)

ypred_gaus_ts=gaus.predict(x_test)
ypred_gaus_tr=gaus.predict(x_train)

accuracy_score(y_test,ypred_gaus_ts)#60
accuracy_score(y_train,ypred_gaus_tr)#58

gaus.coef_
gaus.score
r2_2=r2_score(y_test,ypred_gaus_ts)#15


#multinominal

from sklearn.naive_bayes import MultinomialNB
mul=MultinomialNB()
mul.fit(x_train,y_train)


#
#SVM 

from sklearn.svm import SVC
clf=SVC(random_state=0)
clf.fit(x_train,y_train)

ypred_svc_ts=clf.predict(x_test)
ypred_svc_tr=clf.predict(x_train)

accuracy_score(y_test.values,ypred_svc_ts)
accuracy_score(y_train.values,ypred_svc_tr)
clf.coef_
clf.score

#sgd
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(shuffle=True,random_state=80)
sgd.fit(x_test,y_test)

ypred_sgd_ts=sgd.predict(x_test)
ypred_sgd_tr=sgd.predict(x_train)

accuracy_score(y_test.values,ypred_sgd_ts)#54
accuracy_score(y_train.values,ypred_sgd_tr)#53

sgd.coef_
sgd.score


r2_2=r2_score(y_test,ypred_sgd_ts)#9

print(classification_report(y_test.values,ypred_sgd_ts))












