import numpy as np

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    #identify what I want to return
    TPR = []
    FPR = []
    threshold = []
    
    #making probabilities into a list rather than array
    probabilities = probabilities.tolist()
    
    #make label into list as well
    labels = labels.tolist()
    
    #Number of positive and negative cases
    pos = len([item for item in labels if item == 1])
    neg = len([item for item in labels if item == 0])
    
    #sorting it in increasing order
    prob_sort = sorted(probabilities)
    
    #loop through the probabilities
    for i in xrange(len(prob_sort)):
        threshold.append(prob_sort[i])
        placeholder = []
        for item in prob_sort:
            if item >= threshold[i]:
                placeholder.append(1)
            else:
                placeholder.append(0)
        TPR.append(float(len([item for item, item2 in zip(placeholder,labels) 
                              if (item == item2) and (item2 == 1)]))/pos)
        FPR.append(float(len([item for item, item2 in zip(placeholder,labels) 
                              if (item != item2) and (item2 == 0)]))/neg)
    return TPR, FPR, threshold
