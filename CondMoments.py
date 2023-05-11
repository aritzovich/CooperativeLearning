import numpy as np

class CondMoments(object):
    '''
    Conditional moments of 0th, 1st and 2nd order.

    The basis for the interaction based classifiers with discrete and continuous variables.
    '''

    def __init__(self, n, card, cardY, sets= None, X= None, Y=None):
        # Number of variables, int. The class has index self.n in the dataset D
        self.n= n
        # Cardinality of discrete variables. Their indices are [0,...,len(card)-1]
        self.card= card

        # indices of discrete variables
        self.indCont= set([ind for ind in range(0,len(card)-1)])
        #indices of discrete variables
        self.indDisc= np.nonzero(card!=np.inf)

        self.cardY= cardY


        # initialize the list of sets, list(tuple(set(int))), and the associated list of counts, list(np.array(card[S]))
        if sets is not None:
            #make a copy and transform to tuples

            U,V= self._getUandVfromSets(sets)
            self.initCounts(U,V)
            # learn maximum likelihood statistics
            if X is not None and Y is not None:
                self.learnMoments(X,Y)
        else:
            # Subsets of variables, list(((discInd),(contInd))), e.g., self.U[i][0] discrete vars of i-th supervised stats.
            self.U = None
            self.V = None

        # Moments associated to subsets of cont vars conditioned to discrete vars
        # Supervised moments
        # Indexing: self.M0u[self.U[i]][x[self.U[i][0]]] -> supervised moment 1 of contVars U[i][1] for the configuration
        # of discVars x[self.U[i][0]]
        # M0u[vars][discConfig] -> array[contVars x classConfig]
        self.M0u = None
        # M1u[vars][discConfig] -> array[contVars x classConfig]
        self.M1u = None
        # M2u[vars][discConfig] -> array[contVars x contVars x classConfig]
        self.M2u = None
        # Indexing: self.M0v[self.V[i]][x[self.V[i][0]]] -> unsupervised moment 1 of contVars V[i][1] for the configuration
        # of discVars x[self.U[i][0]]
        # M0u[vars][discConfig] -> array[contVars]
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
                self.U.append((tuple(s for s in S if s!= self.n and s in self.indDisc),tuple(s for s in S if s!= self.n and s in self.indCont)))
            else:
                self.V.append((tuple(s for s in S if s in self.indDisc),tuple(s for s in S if s in self.indCont)))
        return U,V

    def initCounts(self, U= None, V= None, esz= 0):
        '''
        Take the list of set of variables and stores in standard form (tuple(set(int)) where the sets S with the class
        (self.n in S) are stored in self.U (after removing the class variable), and the sets without it are stored in
        self.V

        //TODO lo del equivalent size en continuo: ¿usar como prior la media de los momentos sin condicionar?

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

        #TODO diferenciar continuous de discretos. No hay momentos de orden mayor que 0 en discreto
        if U is None:
            for S in self.U:
                self.M0u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (self.cardY,))
                self.M1u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (self.cardY,len(S[1])))
                self.M2u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (self.cardY,len(S[1]),len(S[1])))

        else:
            self.M0u = {S: np.ones(shape=tuple(self.card[disc] for disc in S[0]) + (self.cardY,)) for S in self.U}

        self.M0u[S] = {S: np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (self.cardY,)) for S in self.U}
        self.M1u[S] = {S: np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (self.cardY, len(S[1]))) for S in self.U}
        self.M2u[S] = {S: np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (self.cardY, len(S[1]), len(S[1]))) for S in self.U}

        if V is None:
            for S in self.V:
                self.M0u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S[0]))
                self.M1u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (len(S[1]),))
                self.M2u[S]= np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (len(S[1]),len(S[1])))
        else:
            self.M0u[S] = {S: np.zeros(shape=tuple(self.card[disc] for disc in S[0])) for S in self.V}
            self.M1u[S] = {S: np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (len(S[1]),)) for S in
                           self.V}
            self.M2u[S] = {
                S: np.zeros(shape=tuple(self.card[disc] for disc in S[0]) + (len(S[1]), len(S[1]))) for S in
                self.V}

        # Avoid problems with the count associated to the empty set of variables
        if ((),()) in self.U: self.Nu[((),())]= np.zeros(shape=self.cardY)
        if ((),()) in self.V: self.Nv[((),())]= np.zeros(shape=1)

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
        stats = ContStats(self.n, self.card, self.cardY)
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
    def maximumLikelihood(self, X,Y, U= None, V= None, esz= 0.0):
        '''
        Learn the moments

        :param X: instances, np.array(num-instances x num features, int)
        :param Y: classes, np.array(num-instances, int)
        :param esz: equivalent sample size
        :return:
        '''

        # Initialize the statistics
        self.initCounts(self.U if U is None else U, self.V if V is None else V , esz)

        # Count the statistics in the data
        m,n= X.shape
        for S in self.U:
            M0u= self.M0u[S]
            M1u= self.M1u[S]
            M2u= self.M2u[S]
            for i in range(m):
                #//TODO comprobar que hay que poner [Y[i]] y no [Y[i],:] o [Y[i],]
                M0u[tuple(X[i,S[0]])][Y[i]] += 1
                M1u[tuple(X[i,S[0]])][Y[i]] += X[i,S[1]]
                M2u[tuple(X[i,S[0]])][Y[i]] += np.dot(X[i,S[1]].reshape(len(S[1]),1), X[i,S[1]].reshape(1,len(S[1])))

        for S in self.V:
            Nv= self.Nv[S]
            for i in range(m):
                Nv[tuple(X[i,S])] += 1

            M0v= self.M0v[S]
            M1v= self.M1v[S]
            M2v= self.M2v[S]
            for i in range(m):
                M0v[tuple(X[i,S[0]])] += 1
                M1v[tuple(X[i,S[0]])] += X[i,S[1]]
                M2v[tuple(X[i,S[0]])] += np.dot(X[i,S[1]].reshape(len(S[1]),1), X[i,S[1]].reshape(1,len(S[1])))

        return

    #//TODO testar
    def maximumWLikelihood(self, X, pY, esz= 0.0):

        # Initialize the statistics
        self.initCounts(esz=esz)

        # Count the statistics in the data
        m, n = X.shape
        for S in self.U:
            M0u = self.M0u[S]
            M1u = self.M1u[S]
            M2u = self.M2u[S]
            for i in range(m):
                M0u[tuple(X[i, S[0]])] += pY[i,:]
                #//TODO ver si cuadra
                M1u[tuple(X[i, S[0]])] += pY[i,:].reshape((self.cardY,1))*  X[i,S[1]].reshape((1,len(S[1])))
                #//TODO ver si cuadra
                M2u[tuple(X[i, S[0]])] += (pY[i,:].reshape((self.cardY,1))
                            *(X[i,S[1]].reshape((len(S[1]),1)) * X[i,S[1]].reshape((1,len(S[1])))).reshape((1,len(S[1])*len(S[1]))).reshape((self.cardY,len(S[1]),len(S[1]))))

        for S in self.V:
            M0v = self.M0v[S]
            M1v = self.M1v[S]
            M2v = self.M2v[S]
            for i in range(m):
                M0v[tuple(X[i, S[0]])] += 1
                #//TODO ver si cuadra
                M1v[tuple(X[i, S[0]])] += X[i,S[1]]
                #//TODO ver si cuadra
                M2v[tuple(X[i, S[0]])] += X[i,S[1]].reshape((len(S[1]),1)) * X[i,S[1]].reshape((1,len(S[1])))

        #for S in self.Nu.keys():
        #    print(str(S) + ":\t" + str(np.sum(self.Nu[S])))








