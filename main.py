# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import Stats as st
import IBC as ibc
import ContStats as cst

def loadSuperisedData(dataName= 'glass.csv', classInd=None, sep=',', maxDiscVals=5, skipHeader= 1):
    '''
    Load a supervised classification data in csv format: each row contains an instance (case), each column corresponds
    to the values of random variable. The values included in each row are separated by a string given by the parameter
    sep, e.g., ",". The column of the class variable is given by classInd, and in case of classInd==None the class
    variable corresponds to the last column.

    A variable is considered discrete when i) the values are strings, or 2) the number of values it takes is smaller or
    equal to maxDiscVals.

    :return: the data set np.array(numCases x numVars), and the cardinality of the variables where continuous have
    np.inf cardinality
    '''

    text = np.genfromtxt(dataName, dtype= np.str, delimiter=sep, skip_header= skipHeader)
    (m, n) = text.shape
    if classInd is None:
        classInd= n-1

    # Determine the nature of the variables (categorical or continuous)
    card = np.zeros(n)
    for i in range(n):
        if i != classInd:
            vals= np.unique(text[:, i])
            if str.isalpha(text[0, i]):
                card[i]= len(vals)

            elif str.isdigit(text[0, i]):  # if all characters in the string are alphabetic or there is at least one character
                if len(vals)<= maxDiscVals:
                    card[i] = len(vals)
                else:
                    card[i] = np.inf
            else:
                if len(vals) <= maxDiscVals:
                    card[i] = len(vals)
                else:
                    card[i] = np.inf
        else:
            card[i]= len(np.unique(text[:,i]))


    data = np.zeros((m,n))
    for i in range(n):
        if card[i] is not np.inf:
            data[:,i] = np.unique(text[:, i], return_inverse=True)[1]
        else:
            data[:,i] = np.array([np.float(x) for x in text[:, i]])


    return data,card



def pruevas_contStat(dataName= 'glass'):

    D,card= loadSuperisedData('./data/'+dataName+'.csv')

    print(str(card)+'\n'+str(D))




def pruebas(structs= ["NB"], domain= "ParSum", ms=[500],ns=[10],seed= 1, r= 2):

    np.random.seed(seed)
    mTest= 1000

    for n in ns:
        for m in ms:
            card = np.ones(n, dtype=int)* r
            cardY = 2

            h = ibc.IBC(card, cardY)

            D = generateData(domain, m, n+1, r= r)
            T = generateData(domain, mTest, n+1, r= r)

            for struct in structs:
                if struct== "NB":
                    h.setBNstruct([[n] for i in range(n)]+[[]])
                elif struct== "TAN":
                    # (0,1), (1,2), ..., (n-2,n-1)
                    h.setBNstruct([[n]] + [[i - 1, n] for i in range(1, n)] + [[]])
                elif struct== "2IBC":
                    h.setKOrderStruct(k=2)

                h.learnMaxLikelihood(D[:,:-1],D[:,-1],esz=0.1)
                print("n= "+str(n)+" m= "+str(m)+" gen-"+struct + "+LL:\t train: "+ str(h.error(D[:,:-1], D[:,-1]))
                      + "\t test: "+ str(h.error(T[:,:-1], T[:,-1])))

                CLL= h.learnCondMaxLikelihood(D[:,:-1],D[:,-1],esz=0.1,max_iter=10)
                #for i in range(3):
                #    CLL= np.append(CLL, h.learnCondMaxLikelihood(D[:,:-1],D[:,-1], stats= h.stats, stats0= h.stats, esz=0.1,max_iter=2))
                print("n= "+str(n)+" m= "+str(m)+" dis-"+struct + "+LL:\t train: "+ str(h.error(D[:,:-1], D[:,-1]))
                      + "\t test: "+ str(h.error(T[:,:-1], T[:,-1])))

                plt.plot(np.array(CLL)/(m*n),label= struct+ " n="+str(n)+" m="+str(m))

    plt.legend(loc="upper left")
    plt.show()


from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
def TMevolution(n= 10, m=100, r= 2, seed= 0):

    np.random.seed(seed)
    mTest= 1000

    card = np.ones(n, dtype=int)* r
    cardY = 2

    # TAN
    h = ibc.IBC(card, cardY)
    h.setBNstruct([[n]] + [[i - 1, n] for i in range(1, n)] + [[]])

    # STANDARD
    D = generateData("ParSum", m, n+1, r= r)
    X= D[:,:-1]
    Y= D[:,-1]
    (CLL,stats)=  h.learnCondMaxLikelihood(D[:,:-1],D[:,-1],esz=0.1,max_iter=10,trace=True)

    # DIFFERENT ORIGIN
    D = generateData("ParSum", m, n+1, r= r)
    stats0= h.stats.emptyCopy()
    stats0.maximumLikelihood(D[:,:-1],D[:,-1])
    (CLL,stats)=  h.learnCondMaxLikelihood(D[:,:-1],D[:,-1],esz=0.1,max_iter=10,trace=True)
    n2= len(CLL2)
    CLL= CLL+ CLL2

    #

    M= np.row_stack([stat.serialize() for stat in stats])


    # Find columns that are linear combinations of other columns
    '''
    R= np.linalg.qr(M)[1]
    np.sum(np.abs(np.linalg.qr(M)[1]), axis=0) aprox 0
    '''
    embedding = MDS(n_components=2)
    M_2d = embedding.fit_transform(M)
    color= (CLL-np.min(CLL))/(np.max(CLL)-np.min(CLL))
    x = M_2d[:, 0]
    y = M_2d[:, 1]
    c = color
    plt.scatter(x, y, s=200, c=c)
    plt.show()

    x = M_2d[:n1,0]
    y = M_2d[:n1, 1]
    plt.plot(x,y,'.-')
    x = M_2d[n1:,0]
    y = M_2d[n1:, 1]
    plt.plot(x,y,'.-')
    plt.show()


def boostrap():
    '''
    TM is a dynamical system that converges to a fixed point of the CLL. Fixing the data, D, the TM starts from max.
    likelihood estimates and converges to the same fixed point.

    u^t+1= u^t-(E^t - u^0)
    u^0= u^t=0: max likel estimates from D
    E^t: max likel estimates from h(·|x,u^t) and D

    Note that in the first iteration of the TM u^1=E^0. This fact plays an important role in the experiment.

    This experiment tries to analyze strategies to scap from the fixed point using bootstrap. There are two approaches:
    1-Restart: use multiple runs from scratch using bootstrap samples, {B}.

    For each bootstrap B:
    - u^0: max likel. estimates from B
    - E^t: max likel estimates from h(·|x,u^t) and B.

    2-Perturb: use multiple boostrap samples iteratively to refine the obtained fixed point

    once the fixed point u^T is reached for a boostrap B, refine the search using the next botstrap B':
    - u^0: max likel estimated from B'
    - u^t=0: u^0- (E^t - u^0)= E^t
    - E^t: max likel estimates from B' h("|x, u^t)

    E^t is the nexus between the fixed point u^T obtained for B and the new TM for B'. In the first iteration of
    the TM, u^1=E^t.

    :return:
    '''

    return

def learningRate():
    '''
    The TM is a dynamical system completely determined by the sample, D.

    u^t+1= u^t - lambda·(E^t - u^0)

    Originally, the learning (updating) rate is lambda= 1. However, depending on the value of lambda different
    dynamical systems are defined.

    1-Analyze the effect of lambda

    For lambda in L get the sequence of statistics U_l, project the higher order statistics into a 2D space and plot
    the evolution of u^t in U^L coloring the point of u^t using CLL(u^t,D).

    :return:
    '''

    return


def landscape():
    '''
    The TM is a dynamical system completely determined by the sample, D.

    u^t+1= u^t - lambda·(E^t - u^0)

    In the first iteration u^1=E^0, the maximum likelihood estimate obtained from D and h(·|x,u^0). This fact plays an
    important role in the experiment.

    1-Analyze the effect neighborhood of u^t

    Using D apply the TM to obtain the sequence of statistics U=(u^1,...,u^T). For each u^t obtain a perturbed version
    u:
    a) u is the max likelihood estimate from h(·|x,u^t) and a boostrap B
    b) u is the max likelihood estimate from a noisy version of h(·|x,u^t) and D

    :return:
    '''
    return


def generateData(domain= "ParSum", m=10, n=5, r= 4):
    if(domain == "parity"):
        p_flip= 0.1
        D= np.random.choice(a=[0,1],size=(m,n))
        D[:,n-1]= (D[:,n-1]+np.random.choice(a=[0,1],size=m,p=[1-p_flip, p_flip]))%2

    elif(domain == "ParSum"):
        D = np.random.choice(a=[i for i in range(r)], size=(m, n))
        fx= sum([(-1)**((D[:,2*i]+D[:,2*i+1])%2) for i in range(int(n/2))])
        py_x= 1/(1+ np.exp(-fx))
        for i in range(m):
            D[i,-1]= np.random.choice(a=[0,1], size= 1, p=[1-py_x[i], py_x[i]])[0]

    return D


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    pruevas_contStat()
#    pruebas()
#    TMevolution(n=10, m=100, r=2, seed=0)

