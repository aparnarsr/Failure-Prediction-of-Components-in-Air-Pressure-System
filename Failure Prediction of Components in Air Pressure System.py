# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:08:52 2019

@author: aparn
"""


import sklearn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.metrics import classification_report, confusion_matrix
import random
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import csv
import math
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier 


#Loading APS Training Dataset

aps_train=pd.read_csv('aps_failure_training_set_SMALLER.csv',delimiter=',')
aps_test=pd.read_csv('aps_failure_test_set.csv',delimiter=',') [1]

#Removing the label column from Dataset and creating a new array with the Labels

aps_train = aps_train.rename(columns = {'class' : 'Label'})
aps_train['Label'] = aps_train.Label.map({'neg':0, 'pos':1})
aps_trainlabel=aps_train.loc[:,aps_train.columns=='Label']
aps_train=aps_train.drop(columns=['Label'])

aps_test = aps_test.rename(columns = {'class' : 'Label'})
aps_test['Label'] = aps_test.Label.map({'neg':0, 'pos':1})
aps_testlabel=aps_test.loc[:,aps_test.columns=='Label']
aps_test=aps_test.drop(columns=['Label'])

#Eliminating columns with more than 70% missing values

aps_train = aps_train.replace(['na'],[np.NaN]) [1]

tot = aps_train.isnull().sum() / len(aps_train)
missing = tot[tot > 0.70].index
aps_train.drop(missing, axis=1, inplace=True)
aps_test.drop(missing, axis=1, inplace=True)

        
#Filling in Missing values in the dataset

#Converting datatype from Dataframe to float

aps_trainlabel = aps_trainlabel.values#astype('float32')
aps_train = aps_train.values#astype('float32')

X_1 = np.empty((0,163))
X_2 = np.empty((0,163))
for i in range(0,aps_trainlabel.shape[0]):
    if(aps_trainlabel[i]==0):
        X_1 = np.append(X_1,[aps_train[i,:]],axis=0)
    else:
        X_2 = np.append(X_2,[aps_train[i,:]],axis=0)
        
imputer = Imputer(missing_values=np.nan, strategy='median',axis=0) [1]

Y_1 = imputer.fit_transform(X_1[:,:])
Y_2 = imputer.fit_transform(X_2[:,:])

X = np.append(Y_1,Y_2,axis =0)
aps_train=X   

aps_test = aps_test.replace(['na'],[np.NaN])
for column in aps_test:
    aps_test[[column]] = imputer.fit_transform(aps_test[[column]])


#aps_train=aps_train.astype('float32') 
#X=X.astype('float32') 

#Standardizing

stdscaler=StandardScaler() [1]
stdscaler.fit(aps_train)
aps_train = stdscaler.transform(aps_train)
stdscaler.fit(aps_test)
aps_test = stdscaler.transform(aps_test)

#Balancing Data

#SMOTE
#
smt = SMOTE() [1]
aps_train, aps_trainlabel = smt.fit_sample(aps_train, aps_trainlabel)

#Feature selection

fd = SelectKBest(score_func=f_classif,k=150) [1]
fd.fit(aps_train,aps_trainlabel)
Xred= fd.transform(aps_train)
Ytestred = fd.transform(aps_test)

#Feature Reduction
#PCA


fred = PCA(n_components = 40) [1]
fred.fit(Xred)
Xred = fred.transform(Xred)
Ytestred = fred.transform(Ytestred)
aps_train = np.array(Xred)
aps_test = np.array(Ytestred)

#Classification


#Multi layer Perceptron

model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 4), random_state=1) [1]
model.fit(aps_train, aps_trainlabel)                         
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=True,
              epsilon=1e-12, hidden_layer_sizes=(20, 4),
              learning_rate='constant', learning_rate_init=0.01,
              max_iter=1000000, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.2, verbose=False, warm_start=False) [1]

label_train_pred = model.predict(aps_train)
acc_train = accuracy_score(aps_trainlabel,label_train_pred)

print("Multi-Layer Perceptron Classifier")
print("Training Accuracy : ",acc_train*100)

label_test_pred = model.predict(aps_test)
acc_test = accuracy_score(aps_testlabel,label_test_pred)
print("Testing Accuracy : ",acc_test*100)
tn, fp, fn, tp=confusion_matrix(aps_testlabel,label_test_pred).ravel() [1]
print(confusion_matrix(aps_testlabel,label_test_pred))
#cost=fn*10+fp*500
cost=fp*10+fn*500
print('cost of the system is:',cost)
roc = roc_auc_score(aps_testlabel,label_test_pred)
f1 = f1_score(aps_testlabel,label_test_pred)
   
print('f1 score',f1)


#SVM
#Cross Validation

c = np.logspace(-3,3,10)
g = np.logspace(-3,3,10)
meanaccsvm = np.zeros((10, 10))
stdvaccsvm = np.zeros((10, 10))
for j in range(0,10):
    print(j)
    for k in range(0,10):
        cf = c[j]        
        gf = g[k]        
        Kfold = StratifiedKFold(n_splits = 10,shuffle = True) [1]
        i=0;
        mean = 0
        stdv = 0
        for train_index,valid_index in Kfold.split(aps_train,aps_trainlabel):
            print("1")
            feature_train_cv,feature_valid_cv = aps_train[train_index],aps_train[valid_index]
            print("2")
            label_train_cv,label_valid_cv = aps_trainlabel[train_index],aps_trainlabel[valid_index]
            print("3")
            model = SVC(C=cf,gamma = gf)
            print("4")
            model.fit(feature_train_cv,label_train_cv)
            print("5")
            label_valid_pred_cv = model.predict(feature_valid_cv)
            print("6")
            acc_valid =  accuracy_score(label_valid_cv,label_valid_pred_cv)
            print("7")
            mean= mean+acc_valid
            print("8")
            stdv = stdv + math.pow(acc_valid,2)
            i=i+1
            print("inside")
        mean = mean/i
        print("outside")
        stdv = math.sqrt((stdv/i)-math.pow(mean,2))
        meanaccsvm[j,k] = mean
        stdvaccsvm[j,k] = stdv


maxacc=0
for j in range(0,10):
    for k in range(0,10):
        if meanaccsvm[j,k]>=maxacc :
            maxacc = meanaccsvm[j,k]
            copt = c[j]
            gopt = g[k]

count=0
minstdv=100
for j in range(0,10):
    for k in range(0,10):
        if meanaccsvm[j,k]==maxacc :
            count=count+1
            if stdvaccsvm[j,k]<=minstdv:
                minstd = stdvaccsvm[j,k]
                copt = c[j]
                gopt = g[k]

model=SVC(gamma=gopt, C=copt)
#model = SVC(gamma=0.01, C=1000)
model.fit(aps_train,aps_trainlabel)
pred_label_train=model.predict(aps_train)
train_error = 1- accuracy_score(aps_trainlabel, pred_label_train)
train_accuracy=accuracy_score(aps_trainlabel, pred_label_train)*100
print("Support Vector Machine Classifier")
print("Train accuracy obtained is", train_accuracy)

pred_label_test=model.predict(aps_test)
test_error = 1- accuracy_score(aps_testlabel, pred_label_test)
test_accuracy=accuracy_score(aps_testlabel, pred_label_test)*100
print("Test accuracy obtained is", test_accuracy)

tn, fp, fn, tp=confusion_matrix(aps_testlabel,pred_label_test).ravel()
print(confusion_matrix(aps_testlabel,pred_label_test))
cost=fp*10+fn*500
print('cost of the system is:',cost)

roc = roc_auc_score(aps_testlabel,pred_label_test)
f1 = f1_score(aps_testlabel,pred_label_test)
print('f1',f1)               


#Naive Bayes
                
class1 = 0
class2 = 0
for i in range(0,aps_trainlabel.shape[0]):
    if(aps_trainlabel[i]==0):
        class1 = class1+1
    else:
        class2 =class2+1              
        
pc1 = class1/(class1+class2)
pc2 = class2/(class1+class2)
pc = np.array([pc1,pc2])
model = GaussianNB(priors=pc)
model.fit(aps_train,aps_trainlabel)
label_train_pred = model.predict(aps_train)
acc_train = accuracy_score(aps_trainlabel,label_train_pred)
print("Naive Bayes Classifier")
print("Training Accuracy : ",acc_train*100)
label_test_pred = model.predict(aps_test)
acc_test = accuracy_score(aps_testlabel,label_test_pred)
print("Testing Accuracy : ",acc_test*100)

roc = roc_auc_score(aps_testlabel,label_test_pred)
f1 = f1_score(aps_testlabel,label_test_pred)
print('f1',f1)

tn, fp, fn, tp=confusion_matrix(aps_testlabel,label_test_pred).ravel() [1]
print(confusion_matrix(aps_testlabel,label_test_pred))
cost=fp*10+fn*500
print('cost of the system is:',cost)

#Random Forest

deprangerf = np.logspace(1,10,10)
meanaccrf = np.zeros((1, 10))
stdvaccrf = np.zeros((1, 10))
for j in range(0,10):
    dep = deprangerf[j]
    Kfold = StratifiedKFold(n_splits = 10,shuffle = True) [1]
    i=0;
    mean = 0
    stdv = 0
    for train_index,valid_index in Kfold.split(aps_train,aps_trainlabel):
        feature_train_cv,feature_valid_cv =aps_train[train_index],aps_train[valid_index]
        label_train_cv,label_valid_cv =aps_trainlabel[train_index],aps_trainlabel[valid_index]
        model = RandomForestClassifier(n_estimators=10,max_depth=dep)
        model.fit(feature_train_cv,label_train_cv)
        label_valid_pred_cv = model.predict(feature_valid_cv)
        acc_valid = accuracy_score(label_valid_cv,label_valid_pred_cv)
        mean= mean+acc_valid
        stdv = stdv + math.pow(acc_valid,2)
        i=i+1
    mean = mean/i
    stdv = math.sqrt((stdv/i)-math.pow(mean,2))
    meanaccrf[0,j] = mean
    stdvaccrf[0,j] = stdv
maxacc=0
for d in range(0,10):
    if meanaccrf[0,d]>=maxacc :
        maxacc = meanaccrf[0,d]
        dep_optimal = deprangerf[d]
count=0
minstdv=100
for d in range(0,10):
    if meanaccrf[0,d]==maxacc :
        count=count+1
        if stdvaccrf[0,d]<=minstdv:
            minstd = stdvaccrf[0,d]
            dep_optimal = deprangerf[d]
model =RandomForestClassifier(n_estimators=10,max_depth=dep_optimal,class_weight='balanced') [1]
model.fit(aps_train,aps_trainlabel)


label_train_pred = model.predict(aps_train)
acc_train = accuracy_score(aps_trainlabel,label_train_pred)*100
print("Random Forest Classifier")
print("Train accuracy obtained is", acc_train)
label_test_pred = model.predict(aps_test)
acc_test = accuracy_score(aps_testlabel,label_test_pred)*100
print("Test accuracy obtained is", acc_test)

tn, fp, fn, tp=confusion_matrix(aps_testlabel,label_test_pred).ravel() [1]
print(confusion_matrix(aps_testlabel,label_test_pred))
cost=fp*10+fn*500
print('cost of the system is:',cost)

roc = roc_auc_score(aps_testlabel,label_test_pred)
f1 = f1_score(aps_testlabel,label_test_pred)
print('f1',f1)
