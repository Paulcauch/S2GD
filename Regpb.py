import numpy as np 
from scipy.linalg import norm 
from scipy.linalg import svdvals

class RegressPb(object):
    """
    Class for L2 regularized Least Squares and Logistic regularized regression

    A: Data matrix (features) nxd
    y: Data vector (labels) n
    n,d: Dimensions of A
    loss: Loss function to be considered in the regression
        'l2': Least-squares loss 
        'logistic11': Logistic loss
    lbda: Regularization parameter
    """

    def __init__(self, A, y,lbda=0,loss='l2',eps_tol = 1e-6):
        self.A = A
        self.y = y
        self.n, self.d = A.shape
        self.loss = loss
        self.lbda = lbda
        self.eps_tol = eps_tol
        self.kappa_val = self.kappa()

    def fun(self, x):
        if self.loss == 'l2':
            return (1/self.n) * norm(self.A.dot(x) - self.y) ** 2  + 1/2 * self.lbda * norm(x) ** 2 
        elif self.loss == 'logistic11':
            yAx = self.y * self.A.dot(x)
            return np.mean(np.log(1. + np.exp(-yAx))) + 1/2 * self.lbda * norm(x) ** 2 
        elif self.loss == 'logistic01':
            expminAx = np.exp(-self.A.dot(x))
            return - np.mean(self.y * np.log(1/(1+expminAx)) + (1-self.y) * np.log(1 - 1 / (1+expminAx))) + 1/2 * self.lbda * norm(x) ** 2 
    
    def f_i(self,x,i):
        if self.loss == 'l2':
            return (self.A[i].dot(x) - self.y[i]) ** 2 + 1/2 * self.lbda * norm(x) ** 2 
        elif self.loss == 'logistic11':
            yAxi = self.y[i] * self.A[i].dot(x)
            return np.log(1. + np.exp(-yAxi)) + 1/2 * self.lbda * norm(x) ** 2 
    
    def grad(self,x):
        if self.loss == 'l2':
            return (2/self.n) * self.A.T.dot(self.A.dot(x) - self.y) + self.lbda * x
        elif self.loss == 'logistic11':
            yAx = self.y * self.A.dot(x)
            aux = 1. / (1. + np.exp(yAx))
            return - (self.A.T).dot(self.y * aux) / self.n + self.lbda * x
    
    def grad_i(self,x,i):
        if self.loss == 'l2':
            return 2 * (self.A[i].dot(x) - self.y[i]) * self.A[i] + self.lbda*x
        elif self.loss == 'logistic11':
            grad = - self.A[i] * self.y[i] / (1. + np.exp(self.y[i]* self.A[i].dot(x))) + self.lbda * x
            return grad
        
    def lipgrad(self):
        if self.loss == 'l2':
            L = (2/self.n) * norm(self.A, ord=2) ** 2  + self.lbda
        elif self.loss == 'logistic11':
            L = norm(self.A, ord=2) ** 2 / (4. * self.n) + self.lbda
        return L
    
    def cvxval(self):
        if self.loss == 'l2':
            s = svdvals(self.A)
            mu = (2/self.n) * min(s)**2 
            return mu + self.lbda
        elif self.loss == 'logistic11':
            return self.lbda
        
    def kappa(self):
        if self.lbda == 0 :
            return 2*self.lipgrad()/self.eps_tol
        else : 
            return self.lipgrad()/self.cvxval()
