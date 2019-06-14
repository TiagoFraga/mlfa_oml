#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:34:12 2019

@author: tiagofraga
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings("ignore")


start_time = time.time()

digits = load_digits()

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)

threshold_train = np.where((y_train ==0) | (y_train==1) | (y_train == 7) | (y_train == 8))
y_train_thres,x_train_thres = y_train[threshold_train],x_train[threshold_train]
threshold_test = np.where((y_test ==0) | (y_test==1) | (y_test == 7) | (y_test == 8))
y_test_thres,x_test_thres = y_test[threshold_test],x_test[threshold_test]



elements = set(y_train_thres)
print(elements)
choice = set(np.random.choice(list(elements),int(len(list(elements))/2),replace=False))
print(choice)
diff = elements.symmetric_difference(choice)
print(diff)

a = np.where((y_train_thres == list(diff)[0]) | (y_train_thres == list(diff)[1]))
y_train_thres_sel = y_train_thres[a]
x_train_thres_sel = x_train_thres[a]


y_train_new_01 = np.where((y_train_thres==0) | (y_train_thres==1),0,1)
y_test_new_01 = np.where((y_test_thres==0) | (y_test_thres==1),0,1)
y_train_new_07 = np.where((y_train_thres==0) | (y_train_thres==7),0,1)
y_test_new_07 = np.where((y_test_thres==0) | (y_test_thres==7),0,1)
y_train_new_08 = np.where((y_train_thres==0) | (y_train_thres==8),0,1)
y_test_new_08 = np.where((y_test_thres==0) | (y_test_thres==8),0,1)

log_reg_01 = LogisticRegression()
log_reg_07 = LogisticRegression()
log_reg_08 = LogisticRegression()
log_reg_01.fit(x_train_thres,y_train_new_01)
log_reg_07.fit(x_train_thres,y_train_new_07)
log_reg_08.fit(x_train_thres,y_train_new_08)

#score = log_reg_01.score(x_test_thres, y_test_new)
#print(score)
a1 = log_reg_01.predict(x_test_thres)
a7 = log_reg_07.predict(x_test_thres)
a8 = log_reg_08.predict(x_test_thres)

n = "07_08"

pred = []
if n == "01_07":
    for m1,m7,m8 in zip(a1,a7,a8):
        if (m1==0):
            if (m7==0):
                pred.append(0)
            else:
                pred.append(1)
        else:
            if (m7==0):
                pred.append(7)
            else:
                pred.append(8)
elif n=="01_08":
    for m1,m7,m8 in zip(a1,a7,a8):
        if (m1==0):
            if (m8==0):
                pred.append(0)
            else:
                pred.append(1)
        else:
            if (m8==0):
                pred.append(8)
            else:
                pred.append(7)
elif n=="07_08":
    for m1,m7,m8 in zip(a1,a7,a8):
        if (m7==0):
            if (m8==0):
                pred.append(0)
            else:
                pred.append(7)
        else:
            if (m8==0):
                pred.append(8)
            else:
                pred.append(1)
#print(pred)
#print(y_test_thres)
unique, counts = np.unique(np.abs(pred-y_test_thres),return_counts=True)
print(counts[0]/np.sum(counts))

finish_time = time.time() - start_time
print("TEMPO:")
print(finish_time)



