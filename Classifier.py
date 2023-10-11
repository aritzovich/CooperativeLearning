import numpy as np

class Classifier(object):

    def __init__(self,n,cardY):
        self.n= n
        self.cardY= cardY

    def fit(self, X, Y, size= 1):
        '''
        Learn the classifier from data

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        size: equivalent sample size
        '''

        self.computeStatistics(X, Y, size)
        self._computeParams(self.stats)

    def getClassProbs(self,X):
        '''
        Get the predicted conditional class distribution for a set of unlabeled instances

        X: unlabeled instances
        '''
        None

    def predict(self,X):

        return np.argmax(self.getClassProbs(X), axis=1)

    def getStats(self):
        '''
        Returns the statistics of the classifier
        '''
        return self.stats

    def setStats(self,stats):
        '''
        Assigns the imput statistics and computes the associated parameters
        '''
        self.stats= stats.copy()

    def copy(self):
        '''
        Returns a copy of the current classifier
        '''
        None

    def _computeParams(self,stats):
        '''
        Compute the parameters of the classifier given the input statistics
        '''
        None

    def computeParams(self):
        self._computeParams(self.stats)

    def computeStatistics(self,X,Y,size= 0):
        '''
        Compute the statistics from the input training set

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        size: equivalent sample size
        '''
        None