#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: kongdeshun
# Date: 2018.8.11



import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit



#Read file
def readFile(filename):
    x_data = []
    y_data = []
    fp = csv.reader(open(filename,'r'))
    for line in fp:
        x_data.append(line[0])
        y_data.append(line[1])
    return np.array(x_data),np.array(y_data)

#Data separation
def trainTestSplit(x_data,y_data):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    rs = ShuffleSplit(n_splits=1, train_size = 0.7, test_size = 0.3, random_state= 0)
    rs.get_n_splits(x_data)

    for train_index,test_index in rs.split(x_data,y_data):

        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

    return X_train,y_train,X_test,y_test

#Base function
def phi(x,power = 10):

    list = []
    for i in range(power):
        list.append(pow(x,i))
    return list

#get alpha and beta
def getParameter(alpha, beta,mat,XtrainSet,YtrainSet):
    while True:
        alpha_Iteration = alpha
        beta_Iteration = beta

        Sn = np.linalg.inv(np.identity(mat.shape[1]) * alpha + beta * np.dot(mat.T,mat))
        Mn = beta * np.dot(Sn,np.dot(mat.T,YtrainSet))
        a,b = np.linalg.eig(beta * np.dot(mat.T, mat))

        Grama = 0.0
        for i in a:
            if str(type(i)) != '<class \'numpy.float64\'>':
                continue
            Grama = Grama + i / (alpha + i)
        alpha = Grama / np.dot(Mn,Mn.T)
        sum = 0.0
        for j in range(len(mat[0])):
            sum = sum + (YtrainSet[j] - np.dot(Mn,mat[j])) ** 2

        beta = (len(XtrainSet) - Grama) / sum

        if abs(alpha_Iteration-alpha) < 0.001 and abs(beta_Iteration - beta) < 0.001:
            return alpha,beta


def evaluate(mat,PHI,y_train):
    list = []
    sum = 0.0
    for i in range(len(PHI)):
        list.append(pow(np.dot(mat, PHI[i]) - y_train[i], 2))
        sum += pow(np.dot(mat, PHI[i]) - y_train[i], 2)
    print("RMSE = ", np.sqrt(sum / len(y_train)))



if __name__ == "__main__":
    x_data,y_data = readFile('data.csv')
    X_train, y_train, X_test, y_test= trainTestSplit(x_data,y_data)
    X_train = X_train.astype('float64')
    y_train = y_train.astype('float64')
    PHI = []
    for x in X_train:
        PHI.append(phi(x))

    PHI = np.array(PHI)
    omega_normal = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, y_train))

    alpha = 0.1
    beta = 9


    alpha,beta = getParameter(alpha,beta,PHI,X_train,y_train)

    Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
    mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, y_train))

    evaluate(omega_normal,PHI,y_train)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X_train, y_train, s = 30, c ='red', marker='s')
    x = np.arange(0, 13.0, 0.1)
    y = np.dot(omega_normal, phi(x))
    ax.plot(x, y, c='black')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()




