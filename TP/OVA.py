#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:32:58 2019

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



finish = False
digits = load_digits()

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)

print("################ Classificação OVO ###########################")
print("### Menu: ###")
print("1- All numbers;")
print("2- Choose numbers;")    
op = input("Pick a option:  ")

while(finish == False ):
    
    if(op == "1"):
        
        start_time = time.time()
        
        OVA=OneVsRestClassifier(LogisticRegressionCV())
        OVA.fit(x_train,y_train)
        predictionsOVA = OVA.predict(x_test)
        scoreOVA = OVA.score(x_test,y_test)
        print(scoreOVA)

        cmOVA = metrics.confusion_matrix(y_test, predictionsOVA)


        plt.figure(figsize=(9,9))
        sns.heatmap(cmOVA, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Accuracy Score: {0}'.format(scoreOVA)
        plt.title(all_sample_title, size = 15);

        finish_time = time.time() - start_time
        print("TEMPO:")
        print(finish_time)
        finish = True

    elif(op == "2"):
        indices_treino = []
        indices_teste = []
        treino_y = []
        treino_x = []
        teste_y = []
        teste_x = []
        
        
        threshold_train = []
        threshold_test = []
        lista = list(map(int, input("Enter the values: ").split()))
        if(len(lista)<10):
            check = True
            for c in lista: 
                if(c > 9):
                    check = False
            
            if(check == True):

                for l in lista:
                   for i, val in enumerate(y_train):
                       if(l == val):
                           indices_treino.append(i)
                    
                
                for i in indices_treino:
                    treino_y.append(y_train[i])
                    treino_x.append(x_train[i])
                
                
                treino_y = np.array(treino_y)
                treino_x = np.array(treino_x)
                print(treino_y.shape)
                print(treino_x.shape)
                
                
                
                for l in lista:
                   for i, val in enumerate(y_test):
                       if(l == val):
                           indices_teste.append(i)
                
                for i in indices_teste:
                    teste_y.append(y_test[i])
                    teste_x.append(x_test[i])
                    
                teste_y = np.array(teste_y)
                teste_x = np.array(teste_x)
                print(teste_y.shape)
                print(teste_x.shape)
                
                
                
            
                start_time = time.time()
                
                
                OVA=OneVsRestClassifier(LogisticRegressionCV())
                OVA.fit(treino_x,treino_y)
                predictionsOVA = OVA.predict(teste_x)
                scoreOVA = OVA.score(teste_x,teste_y)
                print(scoreOVA)

                cmOVA = metrics.confusion_matrix(teste_y, predictionsOVA)


                plt.figure(figsize=(9,9))
                sns.heatmap(cmOVA, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
                plt.ylabel('Actual label');
                plt.xlabel('Predicted label');
                all_sample_title = 'Accuracy Score: {0}'.format(scoreOVA)
                plt.title(all_sample_title, size = 15);
        
                finish_time = time.time() - start_time
                print("TEMPO:")
                print(finish_time)
                
                finish = True
            else:
                print("Error: Enter a invalid number!")
                print("### Menu: ###")
                print("1- All numbers;")
                print("2- Choose numbers;")    
                op = input("Pick a option...")
                
        else:
            print("Error: Too Much Numbers!")
            print("### Menu: ###")
            print("1- All numbers;")
            print("2- Choose numbers;")    
            op = input("Pick a option...")
    
    else:
        print("Error: Pick a valid option")
        print("### Menu: ###")
        print("1- All numbers;")
        print("2- Choose numbers;")    
        op = input("Pick a option...")
    






