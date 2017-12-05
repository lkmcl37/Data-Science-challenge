# -*- mode: Python; coding: utf-8 -*-
#Author: KaMan Leong

import numpy as np
from random import shuffle
import scipy.misc

class MaxEnt():

    def __init__(self, instances, labels):     
          
        self.load_feature(instances, labels)
        self.model_params = np.zeros((len(self.labels), len(self.feature2id)))

    def load_feature(self, train, labels):
        
        self.labels = {}
        self.feature2id = {}
        self.id2label = {}
        
        #mapping the features and labels to ids
        for label in labels:
            size = len(self.labels)
            self.labels[label] = size
            self.id2label[size] = label
                
        for doc in train:
            
            for feat in doc.features():
                if not feat in self.feature2id:
                    self.feature2id[feat] = len(self.feature2id)
                 
            doc.feature_vector = set()
            for feat in doc.features():
                doc.feature_vector.add(self.feature2id[feat])
            
    #transform the features into vectors
    def vectorize(self, instance):
        
        instance.feature_vector = set()
        for feat in instance.features():
            if feat in self.feature2id:
                instance.feature_vector.add(self.feature2id[feat])
        
    def computePosterior(self, instance):
        
        if instance.feature_vector == None:
            self.vectorize(instance)
            
        # compute p(y|x)
        posterior = np.zeros((len(self.labels)))
        
        for label_id in range(len(self.labels)):
            
            weightSum = 0.0
            for feature in instance.feature_vector:    
                weightSum += self.model_params[label_id, feature]
                    
            posterior[label_id] = weightSum
            
        for label_id in range(len(self.labels)):
            posterior[label_id] = np.exp(posterior[label_id] - scipy.misc.logsumexp([posterior[label] for label in range(len(self.labels))])) 
            
        return posterior
                
    #compute the gradient of each batches
    def computeGradient(self, instances):
        
        ep_model = np.zeros((len(self.labels), len(self.feature2id)))
        ep_empirical = np.zeros((len(self.labels), len(self.feature2id)))
        
        for doc in instances:
            
            posterior = self.computePosterior(doc)
            for feat in doc.feature_vector:
                
                ep_empirical[self.labels[doc.label], feat] += 1
                
                for label_id in range(len(self.labels)):
                    ep_model[label_id, feat] += posterior[label_id];
        
        return ep_empirical - ep_model
    
    def train(self, train_instances, dev_instances, learning_rate = 0.01, batch_size=30):
        
        print("Training the Maximum Entropy model..." + '\n')
        
        likelihood = float("inf")
        gradients = np.zeros((len(self.labels), len(self.feature2id)))
       
        cnt = 0
        best = 0.0
        self.likelihood_list = np.zeros(5)
        
        #for early stop
        max_iter = 50
        
        print("batch size: ", batch_size)
        
        #the parameters of the model are all set to zero before entering the first iteration of training.

        while cnt <= max_iter:
            
            print("Iteration no.", cnt)
            cnt += 1
            shuffle(train_instances)
            
            #go through the mini batches
            for start in range(0, len(train_instances), batch_size):
                gradients = self.computeGradient(train_instances[start:start+batch_size])
                
                #parameters = parameters + learning_rate*gradients (of the current mini batch).
                self.model_params += gradients * learning_rate
            
            for count in range(len(self.likelihood_list)-1):
                self.likelihood_list[count] = self.likelihood_list[count+1]
            
            likelihood = self.computeLikelihood(dev_instances)
            self.likelihood_list[len(self.likelihood_list) - 1] = likelihood
        
            decentCount = 0
            
            #To determine whether a model is converged, I maintained a list with size of 5, to keep track of
            #the trend of negative log likelihood. The general idea is, for each iteration we calculate the 
            #negative log likelihood, then remove the first element of the list and push that element to end of the list. 
            #Then we check whether this list is a decreasing sequence, if yes, then the model is convergence, otherwise we continue the training.
        
            for count in range(len(self.likelihood_list)-1):
                if(self.likelihood_list[count] >= self.likelihood_list[count+1]):
                    decentCount+=1
                else:
                    break
                    
            #check the model against the development set
            accuracy = float(sum([100 for ins in dev_instances if self.classify(ins)==ins.label]))/len(dev_instances)
            
            #save the best model
            if accuracy > best:
                best = accuracy
               
            #for monitering the training status
            print("Log likelihood :", likelihood)
            print("Accuracy: ", accuracy)
            
            if(decentCount == (len(self.likelihood_list)-1)):
                break
        
        print("Best dev accuracy", best)
        

    def computeLikelihood(self, instances):
        
        likelihood = -(sum(np.log(self.computePosterior(inst)[self.labels[inst.label]]) for inst in instances))
        return likelihood
    
    def classify(self, instance):
        
        self.vectorize(instance)
        scores = self.computePosterior(instance)
                
        max_num = 0
        label = 0
    
        for idx, score in enumerate(scores):
          
            if score > max_num:
                max_num = score
                label = idx
          
        return self.id2label[label]
