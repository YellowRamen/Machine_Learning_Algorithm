#__author__ = 'Brian Huang'

#Courtsey to Jason since we worked together on some parts

import numpy as np
import cvxopt

class SVM(object):
    def __init__(self,
                 C,
                 kernel,
                 tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.alpha_m = None
        self.support_weights = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.bias = None

        
    '''Please Note: Jason and I worked on this piece individually
    but we compared answers in the end. I suppose that we had the
    same thought process (hence, similarity in codes) since we were
    both just following the instructions'''
    def gram_matrix(self, X):
        #define a n x n matrix filled with zero
        filler = np.zeros((len(X),len(X)))
        #loop through X to get vector and its index
        for i, item in enumerate(X):
            #loop through X to get corresponding vector
            #and its index
            for j, item2 in enumerate(X):
                #perform kernel op. on the vectors
                ele = self.kernel(item, item2)
                #update element wise
                filler[i][j] += ele
        return filler

    def solve(self, X, y):
        """
        This code solves the quadratic system:
                                               solve for x (lagrange multipliers)
                                               min{Lp(x) = x^{T}Px+q^{T}x}
                                               subject to the following constraints:
                                               Gx \coneleq h
                                               Ax = b such that (b-a_i*y_i*<x,X> = 0)
                                               a_i \leq 0
                                               (slack condition)
                                               a_i \leq C

        X: numpy array of dimension (m, n) - predictor variables (features)
        y: numpy array of dimension (1, m) = target (labels)
        """
        m, n = X.shape

        # K is the the gram matrix or kernel <X, X>
        # P is simply the outer product specified in the dual form (a_i * a_j * y_i * y_i * <X, X>)
        # q in this case is a set of dummy variables
        K = self.gram_matrix(X)
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(-1 * np.ones(len(X)))

        # -a_i \leq 0
        # These are dummy variables for the lagrange multipliers, thus we have a diagonal m x m matrix of -1 ones
        # this formalism constrains the lagrange multipliers to always be positive and greater than or equal to 0
        # because of the formulation of the solver, you have to set the dummy variables to be equal to negative 1
        # (cvxopt creates the lagrange multipliers (a_i = x_i) as factors of these during optimization)
        G = cvxopt.matrix(np.diag(np.ones(m) * -1))
        h = cvxopt.matrix(np.zeros(m))

        # a_i \leq c
        # The slack conditions add an additional variable \zeta (1-\zeta) to the equation,  constrained to a value C.
        G_Sk = cvxopt.matrix(np.diag(np.ones(m)))
        h_Sk = cvxopt.matrix(np.ones(m) * self.C)

        # You actually have to write the equation matrix as a side-by-side formulation going into the constraints:
        # [G,slackG](a_i) = [h, slack_h]
        G = cvxopt.matrix(np.vstack((G, G_Sk)))
        h = cvxopt.matrix(np.vstack((h, h_Sk)))

        # these fulfull Ax = b such that (b-a_i*y_i*<x,X> = 0)
        A = cvxopt.matrix(y, (1, m))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def fit(self, X_train, y_train):

        # compute the m lagrange multipliers and return a list
        self.alpha_m = self.solve(X_train, y_train)

        support_vector_indices = self.alpha_m > self.tol

        self.support_weights = self.alpha_m[support_vector_indices]
        self.support_vectors = X_train[support_vector_indices]
        self.support_vector_labels = y_train[support_vector_indices]

        # bias = y_k - \sum z_i y_i  K(x_k, x_i) (this is the error in the prediction)
        # Thus we can just predict an example with bias of zero, and
        # compute the error to get the initial bias.
        self.bias = 0.0

        # literally just a mean of label differences
        self.bias = np.mean(
            [y_k - self._predict(x_k, self.bias, self.support_weights, self.support_vectors, \
                                 self.support_vector_labels, self.kernel)
             for (y_k, x_k) in zip(self.support_vector_labels, self.support_vectors)])

    def predict(self, X):
        return self._predict(X, self.bias, self.support_weights, self.support_vectors,\
                             self.support_vector_labels, self.kernel)

    def _predict(self, X, bias, support_weights, support_vectors, support_vector_labels, kernel):
        """
        This is an internal method used in two different locations. It computes the SVM cost function sum, and thus
        provides prediction labels.
        """
        result = bias

        #looping through support_vectors while accessing each element
        for i, sup in enumerate(support_vectors):
            #performing element wise operation
            k_support = self.kernel(sup, X)
            #updating result
            result += np.sum(k_support * support_vector_labels[i] * support_weights[i])

        return np.sign(result).item()
