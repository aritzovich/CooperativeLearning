import numpy as np

class CondMoments(object):
    '''
    Conditional moments of 0th, 1st and 2nd order.

    The basis for the interaction based classifiers with discrete and continuous variables.

    This is the extension of Stats class by adding the moments of all the continuous variables.

    The subsets of variables describe de subsets of discrete variables selected. The moments of all the continous variables are modeled.

    The data is organized as an np.array where rows correspond to labeled instances and columns to variables. The
    [0,...,len(self.cardY)-1] variables are discrete, [len(self.cardY),...,n-1] are continuous, and index n refers to
    the class variable

    '''

    def __init__(self, n, card, cardY, sets= None, X= None, Y=None):
        '''

        :param n: Number of predictor variables
        :param card: cardinality of the discrete variables
        :param cardY: cardinality of the class variable
        :param sets: subsets of discrete variables (indices)
        :param X: unlebeled instances, num.instances x n
        :param Y: class labels
        '''
        # Number of variables, int.
        # Discrete variables [0,...,len(self.card)-1]
        # Continous variables [len(self.card),...,self.n-1]
        # Class variable self.n
        self.n= n
        # Cardinality of discrete variables. Their indices are [0,...,len(card)-1]
        if card is not None:
            self.card = card
        else:
            self.card= []
        # The number of continous variables
        self.d=  self.n - len(self.card)
        self.indCont= tuple(i for i in range(len(self.card),n))
        # Cardinality of the class variable
        self.cardY= cardY


        # initialize the list of sets, list(tuple(set(int))), and the associated list of counts, list(np.array(card[S]))
        # the sets only include indices to discrete random variables
        if sets is not None:
            #make a copy and transform to tuples

            U,V= self._getUandVfromSets(sets)
            self.initCounts(U,V)
            # learn maximum likelihood statistics
            if X is not None and Y is not None:
                self.learnMoments(X,Y)
        else:
            # Moments associated to subsets of cont vars conditioned to discrete vars
            # Supervised moments
            # Indexing: self.M0u[self.U[i]][x[self.U[i][0]]] -> supervised moment 1 of contVars U[i][1] for the configuration
            # of discVars x[self.U[i][0]]

            # Order 0 moments
            # M0u[vars][discConfig] -> array[classConfig]
            self.M0u = None
            # Order 1 moments,
            # M1u[vars][discConfig] -> array[contVars x classConfig]
            self.M1u = None
            # Order 2 moments
            # M2u[vars][discConfig] -> array[contVars x contVars x classConfig]
            self.M2u = None
            # Indexing: self.M0v[self.V[i]][x[self.V[i][0]]] -> unsupervised moment 1 of contVars V[i][1] for the configuration
            # of discVars x[self.U[i][0]]
            # M0u[vars][discConfig] -> array[1]
            self.M0v = None
            # M1u[vars][discConfig] -> array[contVars]
            self.M1v = None
            # M2u[vars][discConfig] -> array[contVars x contVars]
            self.M2v = None

    def _getUandVfromSets(self, sets):
        # put the tuples in standard format, tuple(set(whatever))
        self.U= list()
        self.V= list()

        for S in sets:
            if self.n in S:
                self.U.append((tuple(s for s in S if s< len(self.card))))
            else:
                self.V.append((tuple(s for s in S if s< len(self.card))))

        return self.U,self.V

    def initCounts(self, U= None, V= None, esz= 1):
        '''
        Take the list of set of variables and stores in standard form (tuple(set(int)) where the sets S with the class
        (self.n in S) are stored in self.U (after removing the class variable), and the sets without it are stored in
        self.V

        //TODO lo del equivalent size en continuo: ¿usar como prior la media de los momentos sin condicionar?
        // ver: https://en.wikipedia.org/wiki/Conjugate_prior

        :param U: set of variables associated to supervised statistics, list(((discVars),(contVars))). If none use self.U
        :param V: set of variables associated to unsupervised statistics, list(((discVars),(contVars))). if none use self.V
        :param esz: equivalent sample size
        :return:
        '''
        # put the tuples in standard format, tuple(set(whatever))
        if U is not None:
            self.U= U
        elif self.U is None:
            self.U= []
        if V is not None:
            self.V= V
        elif self.V is None:
            self.V= []

        #Las primeras len(self.card) son discretas. El resto continuas.
        if U is None:
            for S in self.U:
                self.M0u[S]= np.ones(shape=tuple(self.card[disc] for disc in S) + (self.cardY,)) * esz
                self.M1u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY,self.d))
                self.M2u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY,self.d,self.d))

        else:
            self.M0u = {S: np.ones(shape=tuple(self.card[disc] for disc in S) + (self.cardY,)) * esz for S in self.U}
            self.M1u = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY, self.d)) for S in self.U}
            self.M2u = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY, self.d, self.d)) for S in self.U}


        if V is None:
            for S in self.V:
                self.M0v[S]= np.ones(shape=tuple(self.card[disc] for disc in S)) * esz
                self.M1v[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d,))
                self.M2v[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d,self.d))
        else:
            self.M0v = {S: np.ones(shape=tuple(self.card[disc] for disc in S)) * esz for S in self.V}
            self.M1v = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d,)) for S in self.V}
            self.M2v = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d, self.d)) for S in self.V}

        # Avoid problems with the count associated to the empty set of variables
        if () in self.U:
            self.M0u[()] = np.ones(shape=(self.cardY,)) * esz
            self.M1u[()] = np.zeros(shape=(self.cardY, self.d))
            self.M2u[()] = np.zeros(shape=(self.cardY, self.d, self.d))
        if () in self.V:
            self.M0v[()] = np.ones(shape=1) * esz
            self.M1v[()] = np.zeros(shape=(self.d,))
            self.M2v[()] = np.zeros(shape=(self.d,self.d))

        # Priors towards varianze= 1 cov= 0
        for S in self.U:
            #TODO inspect all configurations of discrete variables, not only the class
            for y in range(self.cardY):
                self.M2u[S][y] = np.diag(np.ones(self.d) * esz)

        return

    def copy(self):
        '''
        Creates a copy of the statistics self.

        :return: A copy of the statistics self.
        '''
        contStats= ContStats(self.n,self.card,self.cardY)
        contStats.initCounts(self.U,self.V)
        for V in self.V:
            contStats.Nv[V]=self.Nv[V].copy()
        for U in self.U:
            contStats.Nu[U]=self.Nu[U].copy()

        return contStats

    def emptyCopy(self):
        '''
        Creates an empty copy of the statistics self.

        :return: An empty copy of the statistics self.
        '''
        stats = CondMoments(self.n, self.card, self.cardY)
        stats.initCounts(self.U, self.V)

        return stats

    def getSampleSize(self):
        '''
        Get the sample size of the statistics
        :return:
        '''

        if bool(self.Nv):
            return np.sum(next(iter(self.Nv.values())))

        if bool(self.Nu):
            return np.sum(next(iter(self.Nu.values())))

        return 0

    #//TODO hacer esto
    def add(self, stats, prop= 1.0):
        for S in stats.U:
            if S in self.U:
                self.Nu[S] += stats.Nu[S] * prop
            else:
                self.Nu.update({S:stats.Nu[S] * prop})

        for S in stats.V:
            if S in self.V:
                self.Nv[S] += stats.Nv[S] * prop
            else:
                self.Nv.update({S: stats.Nv[S] * prop})

    #//TODO hacer esto
    def subtract(self, stats, prop= 1.0):
        for S in stats.U:
            if S in self.U:
                self.Nu[S] -= stats.Nu[S] * prop
            else:
                self.Nu.update({S:-stats.Nu[S]} * prop)

        for S in stats.V:
            if S in self.V:
                self.Nv[S] -= stats.Nv[S] * prop
            else:
                self.Nv.update({S: -stats.Nv[S] * prop})

    #//TODO testar esto
    def update(self, X, pY, ref_stats, lr=1.0):
        '''
        This method update the statistics:

        self = self - lr · (max_likel_stats(X,pY) - len(X)/esz(ref_stats) * ref_stats)

        ref_stats are rescaled to have the sample size of X. Thus, the update does not change the sample size of self

        :param X: Unsupervised data
        :param pY: probability of the class for the samples X
        :param ref_stats: the reference statistics. Usually, they correspond to the max. lik. stat. of (X,Y)
        :param lr: learning rate
        :return: max_likel_stats(X,pY)
        '''

        MWL= self.emptyCopy()
        MWL.maximumWLikelihood(X, pY)
        esz= X.shape[0]
        esz_ref= ref_stats.getSampleSize()

        self.add(ref_stats, prop=lr* esz/esz_ref)
        self.subtract(MWL, prop=lr)

        return MWL

    #//TODO hacer esto
    def min_ratio(self, stats):
        '''
        Subtract the statistics in a proportion that guarantees that the minimum is at least eps

        min(self.stats/stats)

        :return: min(self.stats/stats)
        '''
        # Find maximum proportion
        propU= 1.0
        for S in stats.U:
            if S in self.U:
                ind= self.Nu[S]- stats.Nu[S]<0
                if np.any(ind):
                    propU= np.min([np.min(self.Nu[S][ind]/ stats.Nu[S][ind]),prop])

        propV= 1.0
        for S in stats.V:
            if S in self.V:
                ind = self.Nv[S] - stats.Nv[S] < 0
                if np.any(ind):
                    propV = np.min([np.min(self.Nv[S][ind] / stats.Nv[S][ind]), prop])

        return (propU,propV)

    #TODO testar esto
    def maximumLikelihood(self, X,Y, U= None, V= None, esz= 1.0):
        '''
        Learn the moments

        :param X: instances, np.array(num-instances x num features, int)
        :param Y: classes, np.array(num-instances, int)
        :param esz: equivalent sample size
        :return:
        '''

        # Initialize the statistics
        self.initCounts(self.U if U is None else U, self.V if V is None else V , esz)

        #TODO fix the issue of being Y continous
        Y = np.array(Y,dtype=np.int32)
        discX= np.array(X[:,:len(self.card)],dtype=np.int32)
        contX= X[:,self.indCont]
        # Count the statistics in the data
        m,n= X.shape
        for S in self.U:
            M0u= self.M0u[S]
            M1u= self.M1u[S]
            M2u= self.M2u[S]
            for i in range(m):
                #//TODO comprobar que hay que poner [Y[i]] y no [Y[i],:] o [Y[i],]
                M0u[tuple(discX[i,S])][Y[i]] += 1
                M1u[tuple(discX[i,S])][Y[i]] += contX[i,:]
                M2u[tuple(discX[i,S])][Y[i]] += np.dot(contX[i,:].reshape(self.d,1), contX[i,:].reshape(1,self.d))

        for S in self.V:
            M0v= self.M0v[S]
            M1v= self.M1v[S]
            M2v= self.M2v[S]
            for i in range(m):
                M0v[tuple(discX[i,S])] += 1
                M1v[tuple(discX[i,S])] += contX[i,:]
                M2v[tuple(discX[i,S])] += np.dot(contX[i,:].reshape(self.d,1), contX[i,:].reshape(1,self.d))

        return

    #//TODO testar
    def maximumWLikelihood(self, X, pY, U= None, V= None, esz= 1.0):

        # Initialize the statistics
        self.initCounts(self.U if U is None else U, self.V if V is None else V , esz)

        discX= np.array(X[:,:len(self.card)])
        contX= X[:,self.indCont]
        # Count the statistics in the data
        m, n = X.shape
        for S in self.U:
            M0u = self.M0u[S]
            M1u = self.M1u[S]
            M2u = self.M2u[S]
            for i in range(m):
                for y in range(self.cardY):
                    M0u[tuple(discX[i, S])][y] += pY[i,y]
                    #//TODO ver si cuadra
                    M1u[tuple(discX[i, S])][y] += pY[i,y]* contX[i, :]
                    #//TODO ver si cuadra
                    if 0> np.min(np.diagonal(pY[i,y]* np.dot(contX[i, :].reshape(self.d, 1), contX[i, :].reshape(1, self.d)))):
                        None
                    M2u[tuple(discX[i, S])][y] += pY[i,y]* np.dot(contX[i, :].reshape(self.d, 1), contX[i, :].reshape(1, self.d))

        for S in self.V:
            M0v = self.M0v[S]
            M1v = self.M1v[S]
            M2v = self.M2v[S]
            for i in range(m):
                M0v[tuple(discX[i,S])] += 1
                M1v[tuple(discX[i,S])] += contX[i,:]
                M2v[tuple(discX[i,S])] += np.dot(contX[i,:].reshape(self.d,1), contX[i,:].reshape(1,self.d))

        #for S in self.Nu.keys():
        #    print(str(S) + ":\t" + str(np.sum(self.Nu[S])))








