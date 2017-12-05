# -*- mode: Python; coding: utf-8 -*-
#Author: KaMan Leong

from __future__ import division
from data_preparator import Generate_data
from maxent import MaxEnt

def accuracy(classifier, test):
    correct = [classifier.classify(x) == x.label for x in test]
    return (float(sum(correct)) / len(correct))*100

def get_data():
   
    print("Loading data.....")
    data_loader = Generate_data()
    train, train_labels = data_loader.load('NYPD_7_Major_Felony_Incidents_train.csv')
    test, test_labels = data_loader.load('NYPD_7_Major_Felony_Incidents_test.csv')
    
    print("train size: " , len(train[:-1000]))
    print("dev size: " , len(train[-1000:]))
    print("test size: " , len(test))
    
    #return train and dev set
    return (train[:-1000], train[-1000:], test, train_labels | test_labels)
       
def test_model():
       
    train, dev, test, labels = get_data()
        
    classifier = MaxEnt(train, labels)
    classifier.train(train, dev, 0.001, 100)
    
    print("Test result", accuracy(classifier, test))
    
if __name__ == '__main__':
    test_model()
