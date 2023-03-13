import numpy as np

class Stats(object):
    '''
    Statistics for the interaction-based classifiers for discrete random variables. It is the basis for the distributed
    learning procedures of interaction-based classifiers
    '''

    def __init__(self, n, card, cardY, sets= None, X= None, Y=None):
        # Number of variables, int. The class has index self.n in the dataset D
        self.n= n
        # Cardinality of variables, np.array(int)
        self.card= card

        self.cardY= cardY


        # initialize the list of sets, list(tuple(set(int))), and the associated list of counts, list(np.array(card[S]))
        if sets is not None:
            #make a copy and transform to tuples

            U,V= self._getUandVfromSets(sets)
            self.initCounts(U,V)
            # learn maximum likelihood statistics
            if X is not None and Y is not None:
                self.maximumLikelihood(X,Y)
            else:
                # Counts associated to subsets, dict(set:np.array(double, card[set]))
                self.Nu = None
                self.Nv = None
        else:
            # Subsets of variables, list(set(int))
            self.U = None
            self.V = None
            # Counts associated to subsets, dict(set:np.array(double, card[set]))
            self.Nu = None
            self.Nv = None

    def _getUandVfromSets(self, sets):
        # put the tuples in standard format, tuple(set(whatever))
        self.U= list()
        self.V= list()

        for S in sets:
            if self.n in S:
                self.U.append(tuple(s for s in S if s!= self.n))
            else:
                self.V.append(S)

        return U,V

    def initCounts(self, U= None, V= None, esz= 0):
        '''
        Take the list of set of variables and stores in standard form (tuple(set(int)) where the sets S with the class
        (self.n in S) are stored in self.U (after removing the class variable), and the sets without it are stored in
        self.V


        :param U: set of variables associated to supervised statistics, list(set(int))
        :param V: set of variables associated to unsupervised statistics, list(set(int))
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

        if U is None:
            for S in self.U:
                self.Nu[S]= np.ones(shape=tuple(self.card[s] for s in S) + (self.cardY,)) * esz / (
                                        np.prod([self.card[s] for s in S]) * self.cardY)

        else:
            try:
                self.Nu = {S: np.ones(shape=tuple(self.card[s] for s in S) + (self.cardY,)) * esz / (
                                    np.prod([self.card[s] for s in S]) * self.cardY) for S in self.U}
            except:
                for S in self.U:
                    np.ones(shape=tuple(self.card[s] for s in S) + (self.cardY,)) * esz / (np.prod([self.card[s] for s in S]) * self.cardY)

        if V is None:
            for S in self.V:
                self.Nv[S]= np.ones(shape=tuple(self.card[s] for s in S)) * esz / np.prod([self.card[s] for s in S])
        else:
            self.Nv = {S: np.ones(shape=tuple(self.card[s] for s in S)) * esz /
                                    np.prod([self.card[s] for s in S]) for S in self.V}


        # Avoid problems with the count associated to the empty set of variables
        if () in self.U: self.Nu[()]= np.ones(shape=self.cardY)* esz/self.cardY
        if () in self.V: self.Nv[()]= np.ones(shape=1)* esz

        return

    def random(self, esz= 10, alpha= None, X= None, seed= None):
        '''
        Generate the values of the statistics counts at random. Statistics are consistent under marginalization

        :param esz: equivalent sample size of the statistics
        :param alpha:
        :param X:
        :param seed:
        :return:
        '''

        if seed:
            np.random.seed(seed)
        if not alpha:
            alpha= 1.0
        if not X:
            X= np.zeros((len(self.card), esz))
            for i in range(len(self.card)):
                X[:,i]= np.random.choice(self.card[i], size= esz, p= np.random.dirichlet(np.ones(self.card[i])*alpha))

        self.initCounts()
        esz = X.shape[0]
        pY= np.random.dirichlet(np.ones(self.cardY)*alpha, size= esz)
        self.maximumWLikelihood(X,pY)

    def copy(self):
        '''
        Creates a copy of the statistics self.

        :return: A copy of the statistics self.
        '''
        stats= Stats(self.n,self.card,self.cardY)
        stats.initCounts(self.U,self.V)
        for V in self.V:
            stats.Nv[V]=self.Nv[V].copy()
        for U in self.U:
            stats.Nu[U]=self.Nu[U].copy()

        return stats

    def emptyCopy(self):
        '''
        Creates an empty copy of the statistics self.

        :return: An empty copy of the statistics self.
        '''
        stats = Stats(self.n, self.card, self.cardY)
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

    def update(self, X, pY, ref_stats, lr=1.0, esz= 0):
        '''
        This method update the statistics:

        self = self - lr · (max_likel_stats(X,pY) - ref_stats)

        The implied statistics are scaled to the sample size of self.

        :param X: Unsupervised data
        :param pY: probability of the class for the samples X
        :param ref_stats: the reference statistics
        :param lr: learning rate
        :return: max_likel_stats(X,pY)
        '''

        MWL= self.emptyCopy()
        MWL.maximumWLikelihood(X, pY, esz=esz)
        N_MWL= MWL.getSampleSize()
        N_ref= ref_stats.getSampleSize()
        N= self.getSampleSize()

        self.add(ref_stats, prop=lr*N/N_ref)
        self.subtract(MWL, prop=lr*N/N_MWL)

        return MWL

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
                    propU= np.min([np.min(self.Nu[S][ind]/ stats.Nu[S][ind]),propU])

        propV= 1.0
        for S in stats.V:
            if S in self.V:
                ind = self.Nv[S] - stats.Nv[S] < 0
                if np.any(ind):
                    propV = np.min([np.min(self.Nv[S][ind] / stats.Nv[S][ind]), propV])

        return (propU,propV)

    def maximumLikelihood(self, X,Y, U= None, V= None, esz= 0.0):
        '''
        Learn the maximum likelihood parameters with complete data

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
            Nu= self.Nu[S]
            for i in range(m):
                Nu[tuple(X[i,S])][Y[i]] += 1

        for S in self.V:
            Nv= self.Nv[S]
            for i in range(m):
                Nv[tuple(X[i,S])] += 1


        #for S in self.Nu.keys():
        #    print(str(S) + ":\t" + str(np.sum(self.Nu[S])))

        return

    def maximumWLikelihood(self, X, pY, esz= 0.0):

        # Initialize the statistics
        self.initCounts(esz=esz)

        # Count the statistics in the data
        m, n = X.shape
        for S in self.U:
            Nu = self.Nu[S]
            for i in range(m):
                Nu[tuple(X[i, S])] += pY[i,:]

        for S in self.V:
            Nv = self.Nv[S]
            for i in range(m):
                Nv[tuple(X[i, S])] += 1

        #for S in self.Nu.keys():
        #    print(str(S) + ":\t" + str(np.sum(self.Nu[S])))

    def entropy(self,sets):
        '''
        Computes the entropy of a list of sets of variables
        :param sets:
        :return:
        '''

        H = np.zeros(len(sets))
        for i,S in enumerate(sets):
            if self.n in S:
                S= tuple(s for s in S if s!= self.n)
                if S in self.Nu:
                    N= np.sum(self.Nu[S])
                    H[i]= np.sum(np.log((N/self.Nu(S))**(self.Nu(S)/N)))
                else:
                    H[i]= np.nan
            else:
                S= tuple(S)
                if S in self.Nv:
                    N= np.sum(self.Nv[S])
                    H[i]= np.sum(np.log((N/self.Nv(S))**(self.Nv(S)/N)))
                else:
                    H[i]= np.nan

        return H

    def serialize(self):
        '''
        Creates a np.array(double) with all the supervised statistics

        The last statistics corresponds to the sample size of the supervised statistics

        :return: The concatenation of the flattened supervised statistics
        '''

        return np.concatenate([np.concatenate([Nu.flatten() for Nu in self.Nu.values()]),
                               np.array(np.sum(next(iter(self.Nu.values())))).flatten()])

    def checkConsistency(self):
        '''
        It checks the consistency of the statistics:
        - Non-negative counts
        - Counts has to sum up the same value

        //TODO: consistency of the statistics under marginalization
        :return:
        '''

        consistent= True

        # All the count have to be greater than zero
        for U in self.U:
            if np.any(self.Nu[U]<0):
                print("INCONSISTENT Nu stats: negative counts")
                consistent= False
                break

        for V in self.V:
            if np.any(self.Nv[V]<0):
                print("INCONSISTENT Nv stats: negative counts")
                consistent= False
                break

        # All the counts have to sum the same
        if self.U:
            sum= 0
            for U in self.U:
                if sum> 0:
                    if not np.isclose(np.sum(self.Nu[U]),sum):
                        print("INCONSISTENT Nu stats: different sums")
                        consistent= False
                        break
                else:
                    sum= np.sum(self.Nu[U])

        return consistent

    def forceConsistency(self, esz= None):
        '''
        Forces the consistency of the statistics by replacing negative counts by zero and by having that all the counts
        sum up to the same value.

        //TODO: implement Stats.forceConsistency procedure to check if corrections are beneficial. It is not clear how to guarantee the consistency under marginalization

        :return:
        '''

        return None


def marginalize(N_S, S, R):
    '''
    Marginalize the statistics associated with the set of variables S to obtain the statistics of the set of
    variables R

    Warning: the order of the variables in S and R have to be compatible. Thas is, the relative order of the elements
    in R is preserver in S.

    to be tested: 221125

    :param N_S: Statistics of the variables in the set S
    :param S: The set of variables, tuple(set(int))
    :param R: A subset of variables of S, tuple(set(int))
    :return: The statistics of the variable
    '''

    # find the indices of the variables of S setminus R
    inds= list()
    for iS,s in enumerate(S):
        if s not in R:
            inds.append(iS)

    # marginalize
    N_R= np.sum(N_S,axis=(inds))

    return N_R








