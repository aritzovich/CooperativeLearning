from Classifier import Classifier
from LDA import LDA
from IBC import IBC
from Stats import Stats as sts
import numpy as np
from scipy.stats import multivariate_normal

class NaiveBayes(Classifier):
    '''
    Naive Bayes classifier
    arg max_y p(y) · prod_i p(x_i|y)
    '''

    def __init__(self, n, cardY, card):
        Classifier.__init__(self, n, cardY)
        self.card= card

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
        m,n = X.shape
        py = np.tile(self.py,m).reshape(m,self.cardY)
        for i in range(self.n):
            py *= self.pi_y[i][X[:,i],:]
        #normalize p(y,x) to obtain p(y|x) for each x in X
        py /= np.repeat(np.sum(py,axis=1), self.cardY).reshape((m, self.cardY))

        return py

    def predict(self,X):

        super().predict(X)

    def getStats(self):
        '''
        Returns the statistics of the classifier
        '''
        return super().getStats()

    def setStats(self,stats):
        '''
        Assigns the imput statistics and computes the associated parameters
        '''
        super().setStats(stats)

    def copy(self):
        '''
        Returns a copy of the current classifier
        '''
        copy= NaiveBayes(self.n,self.cardY,self.card)
        copy.setStats(self.getStats())
        copy._computeParams(copy.stats)

        return copy


    def _computeParams(self,stats):
        '''
        Compute the parameters of the classifier given the input statistics
        '''
        self.py= np.array(stats.Nu[tuple()])
        self.py/= np.sum(self.py)

        self.pi_y= [np.array(stats.Nu[(i,)]) for i in range(self.n)]
        for i in range(self.n):
            self.pi_y[i]/= np.tile(np.sum(self.pi_y[i],axis=0),self.card[i]).reshape((self.card[i],self.cardY))


    def computeStatistics(self,X,Y,size= 1):
        '''
        Compute the statistics from the input training set

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        size: equivalent sample size
        '''

        self.stats= sts(self.n, self.card, self.cardY,[tuple([i,self.n]) for i in range(self.n)]+[tuple([self.n])])

        if Y.ndim== 1:
            self.stats.maximumLikelihood(X,Y,esz=size)
        elif Y.ndim== 2:
            self.stats.maximumWLikelihood(X,Y,esz=size)
