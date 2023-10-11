from Classifier import Classifier
from QDA import QDA
import CondMoments as cm
import numpy as np
from scipy.stats import multivariate_normal

class GaussianNaiveBayes(Classifier):
    '''
    Gaussian naive Bayes classifier


    '''

    def __init__(self, n, cardY):
        Classifier.__init__(self, n, cardY)

    def getClassProbs(self,X):
        '''
        Get the predicted conditional class distribution for a set of unlabeled instances

        X: unlabeled instances. Continuos variables.
        '''
        m,n= X.shape
        pY= np.zeros((m,self.cardY))
        for y in range(self.cardY):
            pY[:,y]= self.py[y]* multivariate_normal.pdf(X,mean=self.mu_y[y],cov= np.diag(self.cov_y[y]))

        #normalize p(y,x) to obtain p(y|x) for each x in X
        pY/= np.repeat(np.sum(pY,axis=1), self.cardY).reshape((m, self.cardY))

        #TODO: por alguna razon que se me escapa solo predice las dos primeras clases cuando se aprende por maxima verosimilitud
        return pY

    def _computeParams(self, stats):
        '''
        Compute the parameters of the classifier given the input statistics
        '''

        self.py= np.array(stats.M0y[()])
        self.py/= np.sum(self.py)

        self.mu_y= np.array([stats.M1y[()][y]/stats.M0y[()][y] for y in range(self.cardY)])
        self.cov_y= np.array([np.diagonal(stats.M2y[()][y]) / stats.M0y[()][y] - stats.M1y[()][y]**2 / stats.M0y[()][y]**2 for y in range(self.cardY)])


    def computeStatistics(self,X,Y,size= 1):
        '''
        Compute the statistics from the input training set

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        size: equivalent sample size
        '''

        self.stats= cm.CondMoments(self.n,[],self.cardY,[(self.n,)])

        if Y.ndim== 1:
            self.stats.maximumLikelihood(X,Y,esz=size)
        elif Y.ndim== 2:
            self.stats.maximumWLikelihood(X,Y,esz=size)