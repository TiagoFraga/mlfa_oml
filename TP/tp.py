#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:30:06 2019

@author: tiagofraga
"""

from sklearn.utils.multiclass import unique_labels
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



###################################################################################################
################################# RECOlHA DE DADOS ################################################

digits = load_digits()


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)



threshold_train = np.where((y_train ==0) | (y_train==1) | (y_train == 7) | (y_train == 8))
y_train_thres,x_train_thres = y_train[threshold_train],x_train[threshold_train]
threshold_test = np.where((y_test ==0) | (y_test==1) | (y_test == 7) | (y_test == 8))
y_test_thres,x_test_thres = y_test[threshold_test],x_test[threshold_test]



###################################################################################################
################################# SOFTMAX #########################################################

start_time_SOFT = time.time()

logisticRegr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
logisticRegr.fit(x_train_thres,y_train_thres)
predictions = logisticRegr.predict(x_test_thres)
score = logisticRegr.score(x_test_thres,y_test_thres)

finish_time_SOFT = time.time() - start_time_SOFT

###################################################################################################
################################# OVA #############################################################



start_time_OVA = time.time()

OVA=OneVsRestClassifier(LogisticRegressionCV())
OVA.fit(x_train_thres,y_train_thres)
predictionsOVA = OVA.predict(x_test_thres)
scoreOVA = OVA.score(x_test_thres,y_test_thres)

'''
cmOVA = metrics.confusion_matrix(y_test_thres, predictionsOVA)
plt.figure(figsize=(9,9))
sns.heatmap(cmOVA, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(scoreOVA)
plt.title(all_sample_title, size = 15);
'''

finish_time_OVA = time.time() - start_time_OVA



###################################################################################################
################################# OVO #############################################################


start_time_OVO = time.time()

OVO=OneVsOneClassifier(LogisticRegressionCV())
OVO.fit(x_train_thres,y_train_thres)

predictionsOVO = OVO.predict(x_test_thres)
scoreOVO = OVO.score(x_test_thres,y_test_thres)


'''
cmOVO = metrics.confusion_matrix(y_test_thres, predictionsOVO)
plt.figure(figsize=(9,9))
sns.heatmap(cmOVO, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(scoreOVO)
plt.title(all_sample_title, size = 15);
'''

finish_time_OVO = time.time() - start_time_OVO




###################################################################################################
################################# DICOTOMIA #######################################################

start_time_DICT = time.time()

elements = set(y_train_thres)
#print(elements)
choice = set(np.random.choice(list(elements),int(len(list(elements))/2),replace=False))
#print(choice)
diff = elements.symmetric_difference(choice)
#print(diff)

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

finish_time_DICT = time.time() - start_time_DICT


###################################################################################################
################################# RESULTADOS ######################################################

print("\n")
print("#############################")
print("####### RESULTADOS ##########")
print("#############################")
print("\n")

print("SCORE_SOFTMAX:")
print(score)
print("TEMPO_SOFT:")
print(finish_time_SOFT)

print("\n")

      
print("SCORE_OVA:")
print(scoreOVA)
print("TEMPO_OVA:")
print(finish_time_OVA)

print("\n")
      
print("SCORE_OVO:")
print(scoreOVO)
print("TEMPO_OVO:")
print(finish_time_OVO)

print("\n")
      
print("SCORE_DICT:")
print(counts[0]/np.sum(counts))
print("TEMPO_DICT:")
print(finish_time_DICT)

print("\n")






