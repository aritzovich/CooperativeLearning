from Classifier import Classifier
import CondMoments as cm
import numpy as np
from scipy.stats import multivariate_normal

class LDA(Classifier):
    '''
    Quadratic discriminant analysis
    arg max_y p(y) · |Sigma(y)|^-1/2 · exp{(x-mu(y))^t · Sigma(y)^-1·(x-mu(y))}
    '''

    def __init__(self, n, cardY):
        Classifier.__init__(self, n, cardY)

    def fit(self,X,Y,size= 1):
        '''
        Learn the classifier from data

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        size: equivalent sample size
        '''

        super().fit(X,Y,size)



    def getClassProbs(self,X):
        '''
        Get the predicted conditional class distribution for a set of unlabeled instances

        X: unlabeled instances. Continuos variables.
        '''
        m,n= X.shape
        py= np.zeros((m,self.cardY))
        for y in range(self.cardY):
            py[:,y]= multivariate_normal.pdf(X, mean=self.mu_y[y], cov= self.cov)
        #normalize p(y,x) to obtain p(y|x) for each x in X
        py/= np.repeat(np.sum(py,axis=1), self.cardY).reshape((m, self.cardY))

        return py

    def predict(self,X):

        return super().predict(X)

    def getStats(self):
        '''
        Returns the statistics of the classifier
        '''
        super().getStats()

    def setStats(self,stats):
        '''
        Assigns the imput statistics and computes the associated parameters
        '''
        super.setStats(stats)

    def copy(self):
        '''
        Returns a copy of the current classifier
        '''
        copy= LDA(self.n,self.cardY)
        copy.setStats(self.getStats())
        copy.computeParams(copy.getStats())

        return copy


    def _computeParams(self, stats):
        '''
        Compute the parameters of the classifier given the input statistics
        '''
        key= tuple()

        self.py= np.array(stats.M0u[key])
        self.py/= np.sum(self.py)

        self.mu_y= [np.array(stats.M1u[key][y])/stats.M0u[key][y] for y in range(self.cardY)]

        self.cov= [np.array(stats.M2v[key])/stats.M0v[key]]
        #self.prec_y= [np.linalg.inv(self.cov_y[y]) for y in range(self.cardY)]
        #self.det_y= [np.sqrt(np.linalg.det(self.cov_y[y])) for y in range(self.cardY)]


    def computeStatistics(self,X,Y,size= 1):
        '''
        Compute the statistics from the input training set

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        size: equivalent sample size
        '''

        self.stats= cm.CondMoments(self.n,[],self.cardY,[tuple(n),tuple([i for i in range(self.n+1)]),tuple([i for i in range(self.n)])])

        if Y.ndim== 1:
            self.stats.maximumLikelihood(X,Y,esz=size)
        elif Y.ndim== 2:
            self.stats.maximumWLikelihood(X,Y,esz=size)