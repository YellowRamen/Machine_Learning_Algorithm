from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
import random as r

class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features, features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.features = features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)


    '''Coutersey to Alejandra for helping me with this function. I was having
    trouble with getting random indicies. She helped me build the samples I needed
    to pass into tree.fit()'''
    def build_forest(self, X, y, num_trees, num_samples, num_features):
    # * Return a list of num_trees DecisionTrees.
        forest = []

	# * for each of the num trees
        for i in xrange (num_trees):
	# * create an random selection of the indices of the arrays, sampling
	# with replacement.
            indices = [i for i in xrange(num_samples)]
            indices_sample = r.sample(indices, len(indices)/3)
            X_sample = X[indices_sample]
            y_sample = y[indices_sample]

	# * use these sample indices to select a subset of X and y
	# with the new sample_X and sample_y, build a new tree as a member
    # of the forest and add to the list.
            tree = DecisionTree()
            tree.fit(X_sample, y_sample, self.features)
            forest.append(tree)
        # * Return a list of num_trees DecisionTrees.
        return forest

    def predict(self, X):

        '''
        Return a numpy array of the labels predicted for the given test data.
        '''

        # * Each one of the trees is allowed to predict on the same row of
        # input data. The majority vote is the output of the whole forest.
        # This becomes a single prediction.

        #create a data list to store predictions
        pred = []
        #temperary data storage for manipulation
        temp_pred = []
        #go through each tree and get result
        for tree in self.forest:
            #return individual results from prediction function
            result = tree.predict(X)
            #store data into temp_pred for further processing
            temp_pred.append(result)
        #convert temp_pred into np array
        temp_array = np.asarray(temp_pred)
        #make transpose
        transposed = temp_array.T
        #loop through col
        for col in xrange(transposed.shape[1]):
            #getting the count of occurance of each prediction
            count = Counter(transposed[:,col])
            #append the max key to pred
            pred.append(max(count, key = count.get))
        return np.asarray(pred)

    def score(self, X, y):

        '''
        Return the accuracy of the Random Forest for the given test data.
        '''

        # * In this case you simply compute the accuracy formula as we have
        #defined in class. Compare predicted y to
        # the actual input y.

        #getting results from prediction
        pred_y = self.predict(X)
        #getting the results of how many match there are
        result = [i == j for i, j in zip(pred_y, y)]
        #getting the number of correct predictions
        correct_pred = len([item for item in result if item == True])
        #number of total classes
        n = len(X)

        return (1./n) * correct_pred
