from Classifier import Classifier
from QDA import QDA
import CondMoments as cm
import numpy as np
from scipy.stats import multivariate_normal

class GaussianNaiveBayes(QDA):
    '''
    Linear discriminant analysis
    arg max_y p(y) · prod_i sigma_i(y)^-1/2 · exp{(x_i-mu_i(y)) · sigma_i(y)^-1·(x_i-mu_i(y))}
    '''

    def __init__(self, n, cardY):
        Classifier.__init__(self, n, cardY)

    def getClassProbs(self,X):
        '''
        Get the predicted conditional class distribution for a set of unlabeled instances

        X: unlabeled instances. Continuos variables.
        '''
        m,n= X.shape
        py= np.zeros((m,self.cardY))
        for y in range(self.cardY):
            py[:,y]= self.py[y]* self.py[y]* multivariate_normal.pdf(X,mean=self.mu_y[y],cov= np.diagonal(self.cov_y[y]))

        #normalize p(y,x) to obtain p(y|x) for each x in X
        py/= np.repeat(np.sum(py,axis=1), self.cardY).reshape((m, self.cardY))

        return py