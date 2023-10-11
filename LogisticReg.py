import numpy as np
from CondMoments import CondMoments
from Classifier import Classifier
from scipy.stats import multivariate_normal

class LogisticReg(Classifier):

    def __init__(self,n,cardY):
        self.n= n
        self.cardY= cardY

    def getClassProbs(self,X):
        '''
        Get the predicted conditional class distribution for a set of unlabeled instances

        X: unlabeled instances
        '''

        # p(y|x) propto exp{alpha_y[y] + sum_i x_i*beta_i[y]
        pY = np.exp(self.alpha_y + X.dot(self.beta_y.transpose()))
        pY /= np.repeat(np.sum(pY,axis=1),self.cardY).reshape(pY.shape)
        return pY

    #TODO remove it. It is just used to check the equivalence with logistic regression clasification rule
    def getNbProbs(self,X):

        m,n= X.shape
        pY= np.zeros((m,self.cardY))
        for y in range(self.cardY):
            try:
                pY[:,y]= multivariate_normal.pdf(X, mean=self.mu_y[y], cov= self.cov)* self.py[y]
            except:
                pY[:, y] = multivariate_normal.pdf(X, mean=self.mu_y[y], cov=self.cov) * self.py[y]
        #normalize p(y,x) to obtain p(y|x) for each x in X
        pY/= np.repeat(np.sum(pY,axis=1), self.cardY).reshape((m, self.cardY))

        return pY
    def predict(self,X):

        return np.argmax(self.getClassProbs(X), axis=1)

    def copy(self):
        '''
        Returns a copy of the current classifier
        '''
        None

    def init(self):
        self.beta_y= np.column_stack([np.random.uniform(0,1,size= self.cardY*(self.n-1)).reshape((self.cardY,(self.n-1))),np.zeros(self.cardY)])
        self.alpha_y= np.random.uniform(0,1,size= self.cardY)

    def _computeParams(self,stats):
        '''
        Compute the parameters of the classifier given the input statistics (CondMoments)
        '''

        # Parameters for Gaussian nb under homocedasticity
        m= np.sum(stats.M0y[()])
        self.py= stats.M0y[()]/m
        self.mu_y= stats.M1y[()]/np.repeat(stats.M0y[()], self.n).reshape(stats.M1y[()].shape)
        self.cov= np.diag(np.sum(stats.M2y[()],axis=0))/m - self.py.transpose().dot(self.mu_y**2)

        # dependent term x_i, beta_y_i= (mu_y_i- mu_r_i)/sigma_i^2 and beta_r_i=0
        self.beta_y= (self.mu_y - self.mu_y[self.cardY-1])/self.cov
        # alfa_y= ln p(y)/p(r) + sum_i (mu_r^2-mu_y^2)/2sigma_i^2 and alpha_r= 0
        self.alpha_y = np.log(self.py / self.py[self.cardY - 1]) + np.sum((self.mu_y[self.cardY - 1]**2 - self.mu_y**2) / (2 * self.cov),axis=1)

    def computeParams(self):
        self._computeParams(self.stats)

    def computeStatistics(self,X,Y,size= 0):
        '''
        Compute the statistics from the input training set

        X: unlabeled instances
        Y: class labels. When Y is a matrix, Y is a probabilistic labeling
        size: equivalent sample size
        '''

        self.stats = CondMoments(self.n, np.zeros(0), self.cardY,[(self.n,)], X, Y, size= size)

        if Y.ndim== 1:
            self.stats.maximumLikelihood(X,Y,esz=size)
        elif Y.ndim== 2:
            self.stats.maximumWLikelihood(X,Y,esz=size)