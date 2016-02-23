import numpy as np

def hypothesis(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted percentages (floats between 0 and 1) 
    for the given data with the given coefficients.
    '''
    #implementing math formula of the hypothesis function
    #under the assumption that X and coeffs are arrays
    
    #Just the hypothesis function
    return 1./(1. + np.exp(-1.*(X.dot(coeffs.T))))

def predict(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted values (0 or 1) for the given data with 
    the given coefficients.
    '''
    #This rounds to 0 or 1 in respect to the threshhold
    return hypothesis(X,coeffs).round()

def cost_function(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the value of the cost function for the data with the 
    given coefficients.
    '''
    
    #getting the shape
    n = X.shape[0]
    
    #summation part
    summation = np.sum(y * np.log(hypothesis(X,coeffs)) 
                      + (1-y) * np.log(1-hypothesis(X,coeffs)))
    
    #return the formula of cost function
    return (-1./n) * summation

def cost_regularized(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the value of the cost function with regularization 
    for the data with the given coefficients.
    '''
    
    #size n
    n = X.shape[0]
    
    #naming my lambda
    lamb = 1.0
    
    #calling cost function to get cost value
    cost = cost_function(X,y,coeffs)
    
    #divide and conquer
    term = lamb/(2.0*n)
    
    #actual formula being returned
    return cost + term * np.sum(coeffs**2)

def cost_gradient(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the cost function at the given value 
    for the coeffs. 

    Return an array of the same size as the coeffs array.
    '''
    #Getting the length of the features
    n = X.shape[0]
    
    #Hypothesis data - y data
    new_pt = hypothesis(X,coeffs) - y
    
    #name a new place holder
    betas = []
    
    #looping through each feature for x_ij term
    for j in xrange(X.shape[1]):
        betas.append((1/n) * np.sum(new_pt * X[:,j]))
    
    return np.asarray(betas)
        
def gradient_regularized(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the cost function with regularization 
    at the given value for the coeffs. 

    Return an array of the same size as the coeffs array.
    '''
    
    #Getting the length of the features
    n = float(X.shape[0])
    
    #lambda is named
    lamb = 1.
    
    #Hypothesis data - y data
    new_pt = hypothesis(X,coeffs) - y
    
    #name a new place holder
    betas = []
    
    # identify when j = 0
    betas.append((1/n) * np.sum(new_pt))
    
    #looping through each feature for x_ij term
    for j in xrange(1,X.shape[1]):
        #getting new values
        m = (1/n) * np.sum(new_pt * X[:,j]) + ((lamb/n)*coeffs[j])
        #appending
        betas.append(m)
    
    return np.asarray(betas)
