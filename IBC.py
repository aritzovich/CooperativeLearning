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

    def initStats(self, esz= 0.1):

        self.stats= st.Stats(self.n, self.card, self.cardY)
        self.stats.initCounts(self.expU.keys(), self.expV.keys(), esz= esz)


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

    def learnCondMaxLikelihood(self, X, Y, init_stats= None, ref_stats= None, max_iter= 10, esz= 1.0, lr= 1.0, trace= False):
        '''
        The TM algorithm

          u^t+1= u^t - lr · (E^t-u_0)

        where u^0 are the initial statistics, E^t is the maximum likelihood statistics obtained from X,p(·|X,u^t)
        and u_0 are the reference statistics

        :param X:
        :param Y:
        :param init_stats: if not None, the initial statistics of the model, u^t for t= 0. Used to compute E^t for t= 0.
        :param ref_stats: if not None, the reference statistics that define the likelihood problem
        :param max_iter: maximum iterations of
        :param esz: equivalent sample size
        :param trace: if Trace returns a summary of the execution of the TM False by default.
        :return: the list of the normalized CLL np.array(double). if trace then returns additionally the corresponding
        list of Stats list(Stats)
        '''

        m,n= X.shape

        # The initial statistics u^0. Used in the first update u^1= u^0 - lr · (E^0-u_0)
        if init_stats is not None:
            self.stats= init_stats.copy()
        else:
            # The statistics of N^t=(U^t,V^t). They are used to compute the expectance E_{N^t}[U|x]
            self.learnMaxLikelihood(X,Y,esz=esz)

        # Reference statistics u_0
        # TODO reference statistics has to have the same equivalent sample size than the expectance E^t 
        if ref_stats is None:
            # Reference statistics, u_0. Required for the update U^t= U^{t-1} + U^0 - E_{N^{t-1}}[U|x]
            ref_stats= self.stats.emptyCopy()
            ref_stats.maximumLikelihood(X,Y,esz= esz)

        evolCLL = list()
        if trace:
            evolStats= list()

        for n_iter in range(max_iter):
            # Compute the conditional probability
            pY= self.getClassProbs(X)

            # Compute and store de cond. log. likel.
            evolCLL.append(self.CLL(X,Y,pY= pY))

            #evolCLL.append(np.sum(np.log(pY[:, Y])))
            if trace:
                evolStats.append(self.stats.copy())

            if n_iter>1:
                #Stoping criteria: CLL does not improve
                if evolCLL[-1]<= evolCLL[-2] or not self.stats.checkConsistency():
                    # undo the update of the statistics
                    N= self.stats.getSampleSize()
                    N_MWL= MWL.getSampleSize()
                    self.stats.add(MWL, prop= lr*N/N_MWL)
                    N_0= ref_stats.getSampleSize()
                    self.stats.subtract(ref_stats, prop= lr*N/N_0)

                    del evolCLL[-1]
                    if trace:
                        del evolStats[-1]

                    # Stop
                    break

            MWL= self.stats.update(X,pY,ref_stats,lr= lr, esz= esz)

        if trace:
            return (np.array(evolCLL), evolStats)
        else:
            return np.array(evolCLL)

    def learnMinLogLoss(self, X, Y, init_stats= None, esz= 1.0, max_iter= 10, lr= 1.0, mb_size= None, fixed_mb= False, trace= False, seed= None):

        if mb_size== None:
            mb_size= X.shape[0]

        return self.minibatchTM(X, Y, size= mb_size, init_stats= init_stats, max_iter= max_iter, esz= esz, lr= lr, trace= trace, seed= seed, fixed_mb= fixed_mb)

    def minibatchTM(self, X, Y, size= 1, init_stats= None, max_iter= 10, esz= 1.0, lr= 1.0, trace= False, seed= None, fixed_mb= False):
        '''
        Mini-batch stochastic TM.

        Mini-batch version of the TM algorithm that updates the statistics using randomly selected batches of data

        For size= 1, is the stochastic TM. The stochastic TM has some similarities with Disciminative Frequency
        Estimate of "Su et al. (2008). Discriminative Parameter Learning for Bayesian Networks".

        TODO: learning rate
        TODO: eliminar metodo y reemplazarlo por learnMinLogLoss

        :param X: training instances
        :param Y: the true class of the instances
        :param size: the size of the minibatches
        :param init_stats: if not None, the initial statistics of the model, u^t for t= 0. Used to compute E^t for t= 0.
        :param max_iter: maximum iterations of
        :param esz: equivalent sample size of the uniform prior for the maximum likelihood initial statistics
        :param trace: if Trace returns a summary of the execution of the DEF. False by default.
        :return: the list of the CLL np.array(double). If trace==True then also the corresponding list of Stats list(Stats)
        '''

        m,n= X.shape

        if seed is not None:
            np.random.seed(seed)


        if init_stats is not None:
            # The initial statistics
            self.stats= init_stats.copy()
        else:
            # The statistics of N^t=(U^t,V^t). They are used to compute the expectance E_{N^t}[U|x]
            self.learnMaxLikelihood(X, Y, esz=esz)


        prevStats= self.stats.copy()
        prevCLL= self.CLL(X,Y,normalize=True)

        evolCLL = [prevCLL]
        if trace:
            evolStats= [prevStats]

        mb_inds= getMinibatchInds(m,size)

        # Iterative update: number of iterations over the whole dataset
        for n_iter in range(max_iter* len(mb_inds)):

            # check if at least m instances have been used: "size" number of instances por each iteration
            if (n_iter % len(mb_inds) ==0) and n_iter>0:
                actCLL= self.CLL(X, Y, normalize=True)

                if not fixed_mb:
                    # Get the indices of the minibatch particion
                    mb_inds= getMinibatchInds(m,size)

                # store the cond. log. likel. after visiting all the data
                evolCLL.append(actCLL)
                if trace:
                    evolStats.append(self.stats.copy())

                # Stop condition: monotone increasing of CLL
                if actCLL< prevCLL or not self.stats.checkConsistency():
                    # Restitute previos statistics
                    self.stats = prevStats

                    del evolCLL[-1]
                    if trace:
                        del evolStats[-1]

                    break
                else:
                    # Store the stats of the classifier as long as the CLL increases
                    prevStats = self.stats.copy()

            # Current minibatch
            mb = mb_inds[n_iter % len(mb_inds)]

            # Update the stats
            self.logLossDescent(X[mb, :], Y[mb])

        if trace:
            return (np.array(evolCLL), evolStats)
        else:
            return np.array(evolCLL)

    def logLossDescent(self, X, Y):
        '''
        This method updates self.stats according to the next rule

        self.stats= self.stats + max_likel(X,Y) - max_likel(X,self.p(Y|X))

        :param X: Instances
        :param Y: Labels
        :return: self.p(Y|X)
        '''

        m= X.shape[0]
        pY = self.getClassProbs(X)

        # Update the stats
        for i in range(m):
            delta = np.array([1 - pY[i, y] if y == Y[i] else 0 - pY[i, y] for y in range(self.cardY)])
            for U in self.stats.U:
                self.stats.Nu[U][tuple(X[i, U])] += delta

        return pY

    def getClassProbs(self,X, stats= None):

        if stats:
            IBC_stats= self.stats
            self.stats= stats

        m,n= X.shape
        py= np.zeros(shape=(m,self.cardY))
        for i in range(m):
            py[i,:]= self.getClassProb(X[i,:])

        if stats:
            self.stats= IBC_stats

        return py

    def getClassProb(self,x, stats= None):
        '''
        TODO arreglar lo de las probs negativas y mayores que 1: tiene que ver con TM, stochasticTM y DEF

        :param xy: the instance
        :return:
        '''

        if stats:
            IBC_stats= self.stats
            self.stats= stats

        try:
            py = np.prod(np.row_stack([self.stats.Nu[U][tuple(x[list(U)])] ** self.expU[U] for U in self.expU]),
                         axis=0) * np.prod([self.stats.Nv[V][tuple(x[list(V)])] ** self.expV[V] for V in self.expV])
        except:
            py = np.prod(np.row_stack([self.stats.Nu[U][tuple(x[list(U)])] ** self.expU[U] for U in self.expU]),
                         axis=0) * np.prod([self.stats.Nv[V][tuple(x[list(V)])] ** self.expV[V] for V in self.expV])

        #[tuple(x[list(U)]) for U in self.expU]
        #[U for U in self.expU]

        if stats:
            self.stats= IBC_stats

        py/= np.sum(py)
        return py

    def error(self, X, Y, deterministic= True, stats= None):

        if stats:
            IBC_stats= self.stats
            self.stats= stats

        m,n= X.shape

        if stats:
            self.stats= IBC_stats

        if deterministic:
            return np.sum(np.argmax(self.getClassProbs(X), axis=1) == Y) / m
        else:
            return np.sum(1 - self.getClassProbs(X)[range(m), Y])/m

    def CLL(self, X, Y, pY=None, normalize= True, stats= None):

        if stats:
            IBC_stats= self.stats
            self.stats= stats

        m,n= X.shape
        if pY is None:
            pY= self.getClassProbs(X)

        # avoid zero probabilities
        minProb= 10**-6
        CLL= np.sum(np.log([np.max((pY[i,Y[i]],minProb)) for i in range(m)]))


        if stats:
            self.stats= IBC_stats
        if normalize:
            CLL= CLL/m

        return CLL


# TOOLS FOR WORKING WITH THE STRUCTURE

def getNaiveBayesStruct(n):
    '''
    Creates the naive Bayes structure

    :param n: number of variables
    :return:
    '''
    return [[n] for i in range(n)] + [[]]

def getRandTANStruct(n, ancOrd= None, seed= None):
    '''
    Creates a random tree-augmented naive Bayes structure. The class has index n

    :param n: number of variables
    :param ancOrd: ancestral order of the variables. By default is generated at random
    :param seed: random seed. By default None
    :return:
    '''

    if seed is not None:
        np.random.seed(seed)

    if ancOrd is None:
        andOrd= np.random.permutation(n)

    W= np.random.random((n,n))
    T,w= maximumWeigtedTree(W)
    for i in range(n):
        # Add the class as the parent of each variable
        T[i].append(n)

    # The class has no parent
    T.append([])


def maximumWeigtedTree(W):
    '''
    A symmetric weight matrix W np.array(n x n)

    The method implements Prim's algorithm for finding a maximum weighted tree

    Inefficient implementation: O(n^3) -- related to tetrahedral numbers, n(n+1)*(n+2)/6)
    By using the sort operator: O(n^2 log n)

    Return the maximum weighted tree over the n indices. The tree is directed with 0 index
    as the root.
    '''
    (n,n)= W.shape

    remain = [i for i in range(1,n)]
    added = [0]
    Pa= [list() for i in range(n)]
    added_weight= 0
    while (len(added) < n):
        maximum = -np.inf
        a = -1
        b = -1
        for i in added:
            for j in remain:
                if maximum < W[i,j]:
                    maximum = W[i,j]
                    a = i
                    b = j

        added_weight+= maximum
        Pa[b].append(a)
        added.append(b)
        remain.remove(b)

    return Pa,added_weight

def getMinibatchInds(m, size, seed= None):
    '''
    Return the list of indices for the minibatch particion of a data set of size m into subsets of the given size.

    If m is not multiple of size the last batch contain repeated instances from the other minibatches at random

    TODO: test this procedure

    :param m:
    :param size:
    :param seed:
    :return:
    '''

    if seed:
        np.random.seed(seed)

    randOrder = np.random.permutation(m)

    mb_inds = [np.array([randOrder[i] for i in range(j*size, (j+1)* size)]) for j in range(int(np.floor(m/size)))]
    #if m % size >0:
    #    rem= m-size*int(np.floor(m/size))
    #    mb_inds.append(np.array(list(randOrder[-rem:]) + [randOrder[i] for i in np.random.choice(m-rem, size= size- rem)]))

    return mb_inds







