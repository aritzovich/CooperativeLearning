import numpy as np
import Stats as st
import itertools as itr
from scipy.special import binom

class IBC(object):

    def __init__(self,card,cardY):
        '''

        :param indY: index of the class
        :param card:
        '''

        #self.indVars

        self.cardY= cardY
        self.card= card
        self.n= len(card)
        self.expU= None
        self.expV= None
        self.stats= None

    def copy(self):

        h= IBC(self.card,self.cardY)
        h.n= self.n
        h.expU= self.expU.copy()
        h.expV= self.expV.copy()
        h.stats= self.stats.copy()
        return h

    def setBNstruct(self,Pa):
        '''
        Set a BN structure for the IBC.

        Nota that the stats are not modified, so they could be inconsistent with the structure.

        :param Pa: the parents, list(list(int)), Pa[i][j]: j-th parent of i-th variable
        :return: None
        '''
        self.expU = dict()
        self.expV = dict()
        setn= set([self.n])
        for x, pa in enumerate(Pa):
            pa = set(pa)
            xpa = set(pa).union(set([x]))

            if self.n in xpa:
                # supervised statistics
                xpa= tuple(xpa - setn)
                if xpa in self.expU:
                    self.expU[xpa]+=1
                else:
                    self.expU.update({xpa:1})
            else:
                #unsupervised statistics
                xpa= tuple(xpa)
                if xpa in self.expV:
                    self.expV[xpa]+=1
                else:
                    self.expV.update({xpa:1})
            if self.n in pa:
                #supervised statistics
                pa= tuple(pa- setn)
                if pa in self.expU:
                    self.expU[pa]-= 1
                else:
                    self.expU.update({pa:-1})
            else:
                #unsupervised statistics
                pa= tuple(pa)
                if pa in self.expV:
                    self.expV[pa]-= 1
                else:
                    self.expV.update({pa:-1})

        self.initStats()

        return

    def setKOrderStruct(self,k=2):
        '''
        K-order interaction based classifier

        p(c|x_[n]) propto prod_{S subseteq [n]: |S|<=k}p(x_S,c)^e_|S|
        where
        e_s= sum_{i=0}^{k-s} (-1)^{k-s}·binom(n-s,j)

        :param k:
        :return:
        '''
        self.expU = dict()
        self.expV = dict()
        e_s= self._exp_s(k,self.n-1)
        for s in range(k+1):
            for S in itr.combinations(range(self.n),s):
                self.expU.update({tuple(set(S)):e_s[s]})

        self.initStats()

        return

    def _exp_s(self,k,n):
        '''
        Compute the exponent of statistics U of orders 0,...,k
        :param k:
        :param n:
        :return:
        '''
        e= np.zeros(k+1)
        e[k]=1
        for s in range(k-1,-1,-1):
            e[s]= 1- np.sum([binom(n-s,r-s)*e[r] for r in range(s+1,k+1)])

        return e

    def initStats(self):

        self.stats= st.Stats(self.n, self.card, self.cardY)
        self.stats.initCounts(self.expU.keys(),self.expV.keys())


    def setStats(self,stats):
        '''
        Replace the statistics of the classifier by those contained in stats

        Warning: if stats does not contain all the required statistics the associated statistics is empty

        to be tested 221130

        :param stats: the statistics to parametrize the IBC
        :return:
        '''

        for S in self.stats.U:
            if S in stats.U:
                self.stats.Nu[S]= stats.Nu[S]

        for S in self.stats.V:
            if S in stats.V:
                self.stats.Nv[S]= stats.Nv[S]

    def learnMaxLikelihood(self,X,Y, esz= 0):
        '''

        :param X: the features
        :param Y: the class
        :return:
        '''

        self.initStats()
        self.stats.maximumLikelihood(X, Y, esz=esz)

    def learnCondMaxLikelihood(self, X, Y, stats= None, stats0= None, max_iter= 10, esz= 0, lr= 1.0, trace= False):
        '''
        The TM algorithm

          u^t+1= u^t - lr · (E^t-u_0)

        where u^0 are the initial statistics, E^t is the maximum likelihood statistics obtained from X,p(Y|X,u^t)
        and u_0 are the reference statistics

        :param X:
        :param Y:
        :param stats: if not None, the initial statistics of the model, u^t for t= 0. Used to compute E^t for t= 0.
        :param stats0: if not None, the reference statistics that define the likelihood problem
        :param max_iter: maximum iterations of
        :param esz: equivalent sample size
        :param trace: if Trace returns a summary of the execution of the TM False by default.
        :return: if trace then returns the list of the CLL np.array(double)
        and the corresponding list of Stats list(Stats)
        '''

        m,n= X.shape


        if stats is not None:
            # The initial statistics
            self.stats= stats.copy()
        else:
            # The statistics of N^t=(U^t,V^t). They are used to compute the expectance E_{N^t}[U|x]
            self.learnMaxLikelihood(X,Y,esz=esz)

        if stats0 is None:
            # maximum likelihood statistics. Required for the update U^t= U^{t-1} + U^0 - E_{N^{t-1}}[U|x]
            stats0= self.stats.emptyCopy()
            stats0.maximumLikelihood(X,Y,esz= esz)

        cont= True
        n_iter= 0

        evolCLL = list()
        if trace:
            evolStats= list()

        while(cont and n_iter< max_iter):
            cont= True
            n_iter+= 1

            # Compute the conditional probability
            pY= self.getClassProbs(X)

            # Compute and store de cond. log. likel.
            evolCLL.append(np.sum(np.log(pY[range(m),Y])))
            if trace:
                evolStats.append(self.stats.copy())

            if n_iter>1:
                #Stoping criteria: CLL does not improve
                if evolCLL[-1]<= evolCLL[-2]:
                    cont= False
                    # undo the update of the statistics
                    N= self.stats.getSampleSize()
                    N_MWL= MWL.getSampleSize()
                    self.stats.add(MWL, lr= lr*N/N_MWL)
                    N_0= stats0.getSampleSize()
                    self.stats.subtract(stats0, lr= lr*N/N_0)

                    del evolCLL[-1]
                    if trace:
                        del evolStats[-1]

            MWL= self.stats.update(X,pY,stats0,lr= lr, esz= esz)

        if trace:
            return (np.array(evolCLL), evolStats)
        else:
            return np.array(evolCLL)

    def getClassProbs(self,X):
        m,n= X.shape
        py= np.zeros(shape=(m,self.cardY))
        for i in range(m):
            py[i,:]= self.getClassProb(X[i,:])

        return py

    def getClassProb(self,x):
        '''

        :param xy: the instance
        :return:
        '''

        py = np.prod(np.row_stack([self.stats.Nu[U][tuple(x[list(U)])] ** self.expU[U] for U in self.expU]),
                     axis=0) * np.prod([self.stats.Nv[V][tuple(x[list(V)])] ** self.expV[V] for V in self.expV])

        #[tuple(x[list(U)]) for U in self.expU]
        #[U for U in self.expU]

        py/= np.sum(py)
        return py

    def error(self, X, Y, deterministic= True):

        m,n= X.shape
        if deterministic:
            return np.sum(np.argmax(self.getClassProbs(X), axis=1) == Y) / m
        else:
            return np.sum(1 - self.getClassProbs(X)[range(m), Y])/m

    def condLogLikelihood(self, X, Y):

        m,n= X.shape
        pY= self.getClassProbs(X)
        return np.sum(np.log([pY[i,Y[i]] for i in range(m)]))






