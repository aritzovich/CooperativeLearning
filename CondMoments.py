import numpy as np

class CondMoments(object):
    '''
    Conditional moments of 0th, 1st and 2nd order.

    The basis for the interaction based classifiers with discrete and continuous variables.

    This is the extension of Stats class by adding the moments of all the continuous variables.

    The subsets of variables describe de subsets of discrete variables selected. The moments of all the continous
    variables are modeled.

    The data is organized as an np.array where rows correspond to labeled instances and columns to variables. The
    [0,...,len(self.card)-1] variables are discrete, [len(self.card),...,n-1] are continuous, and index n refers to
    the class variable

    TODO se puede mejorar creando una clase moments donde se especifica el maximo orden. Despues un unico diccionario de
    momentos en lugar de tener tantos diccionarios como momentos.

    '''

    def __init__(self, n, card, cardY, sets= None, X= None, Y=None, size= 0):
        '''
        Constructor

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
            self.initCounts(U, V, esz= size)
            # learn maximum likelihood statistics
            if X is not None and Y is not None:
                self.maximumLikelihood(X, Y, esz= size)
        else:
            # Moments associated to subsets of cont vars conditioned to discrete vars
            # Supervised moments
            # Indexing: self.M0y[self.U[i]][x[self.U[i][0]]] -> supervised moment 1 of contVars U[i][1] for the configuration
            # of discVars x[self.U[i][0]]

            # Order 0 moments
            # M0y[vars][discConfig] -> array[classConfig]
            self.M0y = None
            # Order 1 moments,
            # M1y[vars][discConfig] -> array[contVars x classConfig]
            self.M1y = None
            # Order 2 moments
            # M2y[vars][discConfig] -> array[contVars x contVars x classConfig]
            self.M2y = None
            # Indexing: self.M0[self.V[i]][x[self.V[i][0]]] -> unsupervised moment 1 of contVars V[i][1] for the configuration
            # of discVars x[self.U[i][0]]
            # M0y[vars][discConfig] -> array[1]
            self.M0 = None
            # M1y[vars][discConfig] -> array[contVars]
            self.M1 = None
            # M2y[vars][discConfig] -> array[contVars x contVars]
            self.M2 = None

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
                self.M0y[S]= np.ones(shape=tuple(self.card[disc] for disc in S) + (self.cardY,)) * esz/self.cardY
                self.M1y[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY,self.d))
                self.M2y[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY,self.d,self.d))

        else:
            self.M0y = {S: np.ones(shape=tuple(self.card[disc] for disc in S) + (self.cardY,)) * esz/self.cardY for S in self.U}
            self.M1y = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY, self.d)) for S in self.U}
            self.M2y = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.cardY, self.d, self.d)) for S in self.U}


        if V is None:
            for S in self.V:
                self.M0[S]= np.ones(shape=tuple(self.card[disc] for disc in S)) * esz
                self.M1[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d,))
                self.M2[S]= np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d,self.d))
        else:
            self.M0 = {S: np.ones(shape=tuple(self.card[disc] for disc in S)) * esz for S in self.V}
            self.M1 = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d,)) for S in self.V}
            self.M2 = {S: np.zeros(shape=tuple(self.card[disc] for disc in S) + (self.d, self.d)) for S in self.V}

        # Avoid problems with the count associated to the empty set of variables
        if () in self.U:
            self.M0y[()] = np.ones(shape=(self.cardY,)) * esz/self.cardY
            self.M1y[()] = np.zeros(shape=(self.cardY, self.d))
            self.M2y[()] = np.zeros(shape=(self.cardY, self.d, self.d))
        if () in self.V:
            self.M0[()] = np.ones(shape=1) * esz
            self.M1[()] = np.zeros(shape=(self.d,))
            self.M2[()] = np.zeros(shape=(self.d,self.d))

        # Priors towards varianze= 1 cov= 0
        # TODO inspect all configurations of discrete variables, not only the class. Meterlo en donde se inicializa el momento de orden 2
        for S in self.U:
            for y in range(self.cardY):
                self.M2y[S][y] = np.diag(np.ones(self.d) * esz/self.cardY)
        for S in self.V:
            for y in range(self.cardY):
                self.M2[S] = np.diag(np.ones(self.d) * esz)

        return

    #TODO quitar todo rastro de U y V en CondMoments y en Stats. Reemplazar por Sy y S
    def copy(self):
        '''
        Creates a copy of the statistics self.

        :return: A copy of the statistics self.
        '''
        stats= CondMoments(self.n,self.card,self.cardY)
        stats.initCounts(self.U,self.V)
        for V in self.V:
            stats.M0[V] = self.M0[V].copy()
            stats.M1[V] = self.M1[V].copy()
            stats.M2[V] = self.M2[V].copy()
        for U in self.U:
            stats.M0y[U] = self.M0y[U].copy()
            stats.M1y[U] = self.M1y[U].copy()
            stats.M2y[U] = self.M2y[U].copy()

        return stats

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

        if bool(self.M0):
            return np.sum(next(iter(self.Nv.values())))

        if bool(self.M0y):
            return np.sum(next(iter(self.Nu.values())))

        return 0

    def add(self, stats, prop= 1.0):
        for S in stats.U:
            if S in self.U:
                self.M0y[S] += stats.M0y[S] * prop
                self.M1y[S] += stats.M1y[S] * prop
                self.M2y[S] += stats.M2y[S] * prop
            else:
                self.M0y.update({S:stats.M0y[S] * prop})
                self.M1y.update({S:stats.M1y[S] * prop})
                self.M2y.update({S:stats.M2y[S] * prop})

        for S in stats.V:
            if S in self.V:
                self.M0[S] += stats.M0[S] * prop
                self.M1[S] += stats.M1[S] * prop
                self.M2[S] += stats.M2[S] * prop
            else:
                self.M0.update({S: stats.M0[S] * prop})
                self.M1.update({S: stats.M1[S] * prop})
                self.M2.update({S: stats.M2[S] * prop})

    def subtract(self, stats, prop= 1.0):
        for S in stats.U:
            if S in self.U:
                self.M0y[S] -= stats.M0y[S] * prop
                self.M1y[S] -= stats.M1y[S] * prop
                self.M2y[S] -= stats.M2y[S] * prop
            else:
                self.M0y.update({S:-stats.M0y[S]} * prop)
                self.M1y.update({S:-stats.M1y[S]} * prop)
                self.M2y.update({S:-stats.M2y[S]} * prop)

        for S in stats.V:
            if S in self.V:
                self.M0[S] -= stats.M0[S] * prop
                self.M1[S] -= stats.M1[S] * prop
                self.M2[S] -= stats.M2[S] * prop
            else:
                self.M0.update({S: -stats.M0[S] * prop})
                self.M1.update({S: -stats.M1[S] * prop})
                self.M2.update({S: -stats.M2[S] * prop})

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
            M0y= self.M0y[S]
            M1y= self.M1y[S]
            M2y= self.M2y[S]
            for i in range(m):
                #//TODO comprobar que hay que poner [Y[i]] y no [Y[i],:] o [Y[i],]
                M0y[tuple(discX[i,S])][Y[i]] += 1
                M1y[tuple(discX[i,S])][Y[i]] += contX[i,:]
                M2y[tuple(discX[i,S])][Y[i]] += np.dot(contX[i,:].reshape(self.d,1), contX[i,:].reshape(1,self.d))

        for S in self.V:
            M0= self.M0[S]
            M1= self.M1[S]
            M2= self.M2[S]
            for i in range(m):
                M0[tuple(discX[i,S])] += 1
                M1[tuple(discX[i,S])] += contX[i,:]
                M2[tuple(discX[i,S])] += np.dot(contX[i,:].reshape(self.d,1), contX[i,:].reshape(1,self.d))

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
            M0y = self.M0y[S]
            M1y = self.M1y[S]
            M2y = self.M2y[S]
            for i in range(m):
                for y in range(self.cardY):
                    M0y[tuple(discX[i, S])][y] += pY[i,y]
                    M1y[tuple(discX[i, S])][y] += pY[i,y]* contX[i, :]
                    #//TODO ver si cuadra
                    M2y[tuple(discX[i, S])][y] += pY[i,y]* np.dot(contX[i, :].reshape(self.d, 1), contX[i, :].reshape(1, self.d))

            #TODO Alternativa eficiente:
            #M1y[y]= np.dot(pY[:,y].T,X)

        for S in self.V:
            M0 = self.M0[S]
            M1 = self.M1[S]
            M2 = self.M2[S]
            for i in range(m):
                M0[tuple(discX[i,S])] += 1
                M1[tuple(discX[i,S])] += contX[i,:]
                M2[tuple(discX[i,S])] += np.dot(contX[i,:].reshape(self.d,1), contX[i,:].reshape(1,self.d))

        #for S in self.Nu.keys():
        #    print(str(S) + ":\t" + str(np.sum(self.Nu[S])))








