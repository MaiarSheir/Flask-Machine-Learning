# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 18:35:32 2019

@author: Maiar
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.tree import DecisionTreeClassifier #DTree
from sklearn.svm import SVC #SVM
from pandas import Series
from flask import request
from flask import Flask


app = Flask(__name__)

@app.route("/")
def C ():
    
    Alg=request.args.get('algorithm')
# Reading Dataset and preprocessing    
    df= pd.read_csv("data.csv")
    df=df[["SeniorCitizen","Dependents","tenure","PhoneService","MultipleLines",
       "InternetService","StreamingTV",
       "StreamingMovies","Contract","TotalCharges","Churn"]]
    
    df=df.dropna()

    encs = {}
    for i in df.columns:
        if df[i].dtypes == 'object':
           encs[i] = LabelEncoder()
           df[i] = encs[i].fit_transform(df[i])
    
    X = df.iloc[:,0 :10].values
    y = df.iloc[:, 10].values   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Getting new data input
    SeniorCitizen=int(request.args.get('SeniorCitizen'))
    Dependents=request.args.get('Dependents')
    if Dependents == "Yes":
       Dependents = 1
    else:
       Dependents = 0
       
    tenure=float(request.args.get('tenure'))
    PhoneService=request.args.get('PhoneService')
    if PhoneService == "Yes":
       PhoneService = 1
    else:
       PhoneService= 0
    MultipleLines=request.args.get('MultipleLines')
    if MultipleLines == "Yes":
       MultipleLines = 1
    else:
       MultipleLines = 0
    InternetService=request.args.get('InternetService')
    if InternetService == "DSL":
       InternetService = 0
    elif InternetService == "Fiber optic":
       InternetService = 1
    else: 
       InternetService = 2 
    StreamingTV=request.args.get('StreamingTV')
    if StreamingTV == "Yes":
       StreamingTV = 1
    else:
       StreamingTV = 0
    StreamingMovies=request.args.get('StreamingMovies')
    if StreamingMovies == "Yes":
       StreamingMovies = 1
    else:
       StreamingMovies = 0
    Contract=request.args.get('Contract')
    if Contract == "Month-to-month":
       Contract = 0
    elif Contract == "One year":
       Contract = 1
    elif Contract == "Two year": 
       Contract = 2
    
    TotalCharges=float(request.args.get('TotalCharges'))   
    
# Algorithms

    if Alg == "knn":
        
       sc_knn = StandardScaler()

       X_train = sc_knn.fit_transform(X_train)
       X_test = sc_knn.transform(X_test)

       classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 3)
       classifier.fit(X_train, y_train)
       
       y_pred_new=classifier.predict(np.array([SeniorCitizen,Dependents,tenure,PhoneService,MultipleLines,
                                                    InternetService,StreamingTV,StreamingMovies,Contract,TotalCharges]).reshape(1,-1))[0]
# Accuracy
       y_pred= classifier.predict(X_test)
       Acc=(accuracy_score(y_test, y_pred))*100
       cm = confusion_matrix(y_test, y_pred)
       
    
    elif Alg == "LogisticRegression" :
       sc_LR = StandardScaler()

       X_train = sc_LR.fit_transform(X_train)
       X_test = sc_LR.transform(X_test)

       classifier = LogisticRegression(random_state = 0)
       classifier.fit(X_train, y_train)
# Accuracy
       y_pred= classifier.predict(X_test)
       
       y_pred_new=classifier.predict(np.array([SeniorCitizen,Dependents,tenure,PhoneService,MultipleLines,
                                                     InternetService,StreamingTV,StreamingMovies,Contract,TotalCharges]).reshape(1,-1))[0]
       
       Acc=accuracy_score(y_test, y_pred)
       cm = confusion_matrix(y_test, y_pred)
       
       
    elif Alg == "NaiveBayes" :
       sc_LR = StandardScaler() 
       classifier = GaussianNB()
       classifier.fit(X_train, y_train) 
# Accuracy 
       y_pred= classifier.predict(X_test)
       y_pred_new=classifier.predict(np.array([SeniorCitizen,Dependents,tenure,PhoneService,MultipleLines,
                                                     InternetService,StreamingTV,StreamingMovies,Contract,TotalCharges]).reshape(1,-1))[0]
       Acc=(accuracy_score(y_test, y_pred))*100
       cm = confusion_matrix(y_test, y_pred)
       
    elif Alg=="SVM":
       sc_LR = StandardScaler() 
       classifier = SVC(kernel = 'rbf', random_state = 0)
       classifier.fit(X_train, y_train)
# Accuracy
       y_pred= classifier.predict(X_test)
       y_pred_new=classifier.predict(np.array([SeniorCitizen,Dependents,tenure,PhoneService,MultipleLines,
                                                     InternetService,StreamingTV,StreamingMovies,Contract,TotalCharges]).reshape(1,-1))[0]
       Acc=(accuracy_score(y_test, y_pred))*100
       cm = confusion_matrix(y_test, y_pred)
       
    else :
       sc_LR = StandardScaler() 
       classifier = DecisionTreeClassifier(random_state=0)
       classifier.fit(X_train, y_train)
# Accuracy
       y_pred= classifier.predict(X_test)
       y_pred_new=classifier.predict(np.array([SeniorCitizen,Dependents,tenure,PhoneService,MultipleLines,
                                                     InternetService,StreamingTV,StreamingMovies,Contract,TotalCharges]).reshape(1,-1))[0]
       Acc=(accuracy_score(y_test, y_pred))*100
       cm = confusion_matrix(y_test, y_pred)
       

 #Output
    return "<h3> The Accuracy of the model : "+str(Acc)+" </h3>\
         <h3> Confusion Matrix of the model : "+str(cm)+"</h3>\
         <h3> Churned or Not ! : "+str(y_pred_new)+"</h3>"
    
    
    
if __name__=='__main__':
    app.run(debug=True)
    
    
    
    
    
    
    