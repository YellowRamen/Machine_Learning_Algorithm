__author__ = 'Brian_Huang'


import numpy as np
from collections import defaultdict as dd

class NaiveBayes(object):

    def __init__(self, alpha=1):
        self.prior = {}
        self.per_feature_per_label = {}
        self.feature_sum_per_label = {}
        self.likelihood = {}
        self.posterior = {}
        self.alpha = alpha
        self.p = None
        
    def compute_prior(self, y):
        #Classifying spam
        for spam in y:
            if spam in self.prior:
                self.prior[spam] += 1.
            else:
                self.prior[spam] = 1.

    #Creating formula method for easier management
    def formula(self, Syi, alpha, Sy, p):
            return (Syi+alpha) / (Sy+(alpha*p))
                
    def compute_likelihood(self, X, y): 
        '''Note: Sy,i is self.per_feature_per_label
        and Sy is self.feature_sum_per_label. First
        loop goes through to get Sy,i and Sy'''
        
        #Making this a defaultdict for easy data structure
        self.per_feature_per_label = dd(list)
        for i, j in zip(X,y):
            #getting Sy,i
            if j not in self.per_feature_per_label:
                self.per_feature_per_label[j] = 0.
            self.per_feature_per_label[j] += i
            #getting Sy
            if j in self.per_feature_per_label:
                self.feature_sum_per_label[j] = sum(self.per_feature_per_label[j])
        
        #p is the number of features
        p = len(self.per_feature_per_label)
        
        #creating data structure for likelihood matrix
        self.likelihood = dd(list)
        
        #loop through each y_value (0, 1)
        for y_value in self.feature_sum_per_label:
            #if no y is stored in likelihood, we set it to 0
            #print self.likelihood[y_value]
            if y_value not in self.likelihood:
                self.likelihood[y_value] = []
            #else, we store the likehood probabilities (from formula)
            self.likelihood[y_value].append(self.formula(np.asarray(self.per_feature_per_label[y_value]).T,
                                                    self.alpha,
                                                    np.asarray(self.feature_sum_per_label[y_value]),
                                                    p))
        
        '''This piece of code restructured the data
        in the likelihood dictionary. My fault for not
        storing it right the first time...'''
        x1 = []
        x2 = []
        
        for k,v in self.likelihood.items():
            if k == 0.0:
                #loop through key to get array
                for item in v:
                    #loop through element of array
                    for ele in item:
                        #append elements individually
                        x1.append(ele)
            if k == 1.0:
                for item in v:
                    for ele in item:
                        x2.append(ele)
        #convert new list into array to avoid double lists
        x1, x2 = np.asarray(x1), np.asarray(x2)
        
        #updating the dictionary
        self.likelihood[0] = x1
        self.likelihood[1] = x2
        
    def fit(self, X, y):
        self.p = X.shape[1]
        self.compute_prior(y)
        self.compute_likelihood(X, y)
        
        
    def predict(self, X):
        classification = []
        for ele in X:
            lst = []
            for key in self.prior:
                prob = np.sum(ele*np.log(self.likelihood.get(key)))+np.log((self.prior[key])/np.sum(self.prior.values()))
                lst.append(prob)
            classification.append(np.argmax(lst))
        return classification
        
    def score(self, X, y):
        diff = self.predict(X) - y
        score = 0.
        for item in diff:
            score += np.absolute(item)
        return 1.-score/len(X)