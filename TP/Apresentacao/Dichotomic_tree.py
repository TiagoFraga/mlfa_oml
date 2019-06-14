import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.datasets import load_digits
import os
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
from time import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import time

np.set_printoptions(suppress=True)

# x_train = np.array([[5,5],[6,5],[5,6],[-5,-5],[-4,-5],[-6,-4],[5,4],[3,7],[5,6],[-4,5],[-5,6],[-6,4]])
# y_train = np.array([1,1,1,2,2,2,4,4,4,6,6,6])
# x_test = np.array([[-2,1],[2,2],[-6,-14]])
# y_test = np.array([6,1,2])

#X = np.array([[1,5],[4,2],[2,3],[5,2],[5,1],[7,4],[2,4],[2,5],[1,3],[4,2],[1,3],[8,8],[6,2]])
#Y = np.array([3,1,2,0,2,0,4,5,4,6,7,7,7])
#X_test = np.array([[5,5],[6,6]])

finish = False
digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


def combinacoes(lst):
    # lst é a lista onde vão ser encontradas as combinações
    # r é o tamanho do primeiro bloco
    r = int(np.ceil(len(lst)/2))
    lista = []
    for m in list(combinations(lst,r)):
        complementar = set(lst).symmetric_difference(set(m))
        m = set(m)
        lista.append((m,complementar))
    if len(lista)%2 == 0:
        lista = lista[0:int(len(lista)/2)]
    return lista

def dataset_to_groups(dataset_labels,groups):
    # dataset_labels é o y_train
    # groups é uma lista de dois grupos
    s1,s2 = set(groups[0]),set(groups[1])
    y = np.zeros_like(dataset_labels)
    for i in s1:
        y[dataset_labels==i]=0
    for i in s2:
        y[dataset_labels==i] = 1
    return y

def find_best_classifier(X,Y):
    elements = set(Y)
    comb = combinacoes(elements)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    if len(set(y_train))==1:
        x_train = X
        y_train = Y
        x_test = None
        y_test = None

    max_score = -1
    for m in comb:
        y_train_norm = dataset_to_groups(y_train,m)
        y_test_norm = dataset_to_groups(y_test,m)
        logistic = LogisticRegression()
        logistic.fit(x_train,y_train_norm)
        if x_test is None:
            score = 0
        else:
            y_pred = logistic.predict(x_test)
            score = logistic.score(x_test,y_test_norm)
        if max_score < score:
            max_score = score
            Y_train = dataset_to_groups(Y,m)
            logistic2 = LogisticRegression()
            logistic2.fit(X,Y_train)
            classifier = logistic2
            groups = m
    return classifier,groups

def restrict_data(X,Y,group):
    mute = np.in1d(Y,list(group))
    x_local = X[mute]
    y_local = Y[mute]
    return x_local,y_local


class Node:
    def __init__(self,classifier=None,groups=None):
        self.left = None
        self.right = None
        self.classifier = classifier
        self.groups = groups
    def __str__(self):
        return str(self.groups[0]) + ',' + str(self.groups[1])

class LogisticTree:
    def __init__(self):
        self.root = None
        self.labels = None

    def _insert_node(self,node,X_local,Y_local):
        # o que acontece se o lado esquerdo tem mais do que um grupo
        if len(node.groups[0]) > 1:
            x_local,y_local = restrict_data(X_local,Y_local,node.groups[0])
            classifier,groups_ = find_best_classifier(x_local,y_local)
            new_node = Node(classifier=classifier,groups=groups_)
            node.left=new_node
            #print(new_node)
            self._insert_node(node.left,x_local,y_local)

        # o que acontece se o lado direito tem mais do que um grupo
        if len(node.groups[1]) > 1:
            x_local,y_local = restrict_data(X_local,Y_local,node.groups[1])
            classifier,groups_ = find_best_classifier(x_local,y_local)
            new_node = Node(classifier=classifier,groups=groups_)
            node.right=new_node
            #print(new_node)
            self._insert_node(node.right,x_local,y_local)

    def fit(self,X,Y):
        self.labels = set(Y)
        classifier,groups_ = find_best_classifier(X,Y)
        new_node = Node(classifier=classifier,groups=groups_)
        self.root = new_node
        #print(new_node)
        self._insert_node(self.root,X,Y)

    def _predict(self,node,X_local):
        Y_pred = node.classifier.predict(X_local) # vector de 0's e 1's
        mute = np.full(len(X_local),np.nan)

        x_test_left = X_local[Y_pred==0]
        x_test_right = X_local[Y_pred==1]


        if (node.left is not None) & (len(x_test_left) != 0):
            mute[Y_pred==0] = self._predict(node.left,x_test_left)
        else:
            mute[Y_pred==0] = list(node.groups[0])[0]
        if (node.right is not None) & (len(x_test_right) !=0):
            mute[Y_pred==1] = self._predict(node.right,x_test_right)
        else:
            mute[Y_pred==1] = list(node.groups[1])[0]
        return mute

    def predict(self,X_test):
        mute = self._predict(self.root,X_test).astype(int)
        return mute

    def score(self,X_test,Y_test):
        if len(X_test) != len(Y_test):
            print('Função score(): tamanho de X_test tem de ser igual a tamanho de Y_test')
            return
        Y_predicted = self.predict(X_test)
        sum = np.sum(Y_predicted==Y_test)
        print(sum)
        print(len(Y_test))
        score_ = sum/len(Y_test)
        return score_

print("################ Classificação DICHOTOMIC ###########################")
print("### Menu: ###")
print("1- All numbers;")
print("2- Choose numbers;")
op = input("Pick an option:  ")

while(finish == False ):

    if(op == "1"):
        start_time = time.time()
        #first = time()
        a = LogisticTree()
        a.fit(x_train,y_train)

        #second = time()
        pred = a.predict(x_test)

        #third = time()
        score = a.score(x_test,y_test)

        #fourth = time()

        #print(second-first)
        #print(third-second)
        #print(fourth-third)
        finish_time = time.time() - start_time
        print("TEMPO:")
        print(finish_time)

        print("SCORE_DICHOTOMIC:")
        print(score)

        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15);
        plt.show()

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
                #first = time()
                a = LogisticTree()
                a.fit(treino_x,treino_y)

                #second = time()
                pred = a.predict(teste_x)

                #third = time()

                score = a.score(teste_x,teste_y)

                #fourth = time()

                #print(second-first)
                #print(third-second)
                #print(fourth-third)
                finish_time = time.time() - start_time
                print("TEMPO:")
                print(finish_time)

                print("SCORE_DICHOTOMIC:")
                print(score)

                cm = confusion_matrix(teste_y, pred)
                plt.figure(figsize=(9,9))
                sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
                plt.ylabel('Actual label');
                plt.xlabel('Predicted label');
                valor = 0.5
                lista_valor = []
                lista_valor.append(valor)
                for i in range(len(lista)-1):
                    valor = valor + 1
                    lista_valor.append(valor)
                plt.xticks(lista_valor,lista)
                plt.yticks(lista_valor,lista)
                all_sample_title = 'Accuracy Score: {0}'.format(score)
                plt.title(all_sample_title, size = 15);
                plt.show()

                finish = True
