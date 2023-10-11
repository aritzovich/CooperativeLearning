# This is a sample Python script.
import time

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas
import sklearn as sk

import IBC
import Stats as sts
import IBC as ibc
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sbn

import Utils
import Utils as utl



def TM_VS_DEF(dataName= 'iris', sizes=[20,35,75,140], numIter= 10, num_rep=10):

    D,card= utl.loadSupervisedData(dataName= './data/'+dataName+'.csv',skipHeader=1, bins=3)
    n= len(card)-1
    cardY= card[-1]
    card= card[:-1]
    m= D.shape[0]
    X= D[:,:-1]
    Y= D[:,-1]

    df= pandas.DataFrame(columns=['seed','data','$m$','method',"n_iter", 'score', 'value'])

    for seed in range(num_rep):
        np.random.seed(seed)
        D = D[np.random.permutation(m),:]

        for size in sizes:
            X = D[:size, :-1]
            Y = D[:size, -1]

            nb= IBC.getNaiveBayesStruct(n)

            h= ibc.IBC(card,cardY)
            h.setBNstruct(nb)
            start= time.localtime()

            CLL= h.learnCondMaxLikelihood(X= X, Y= Y, max_iter= numIter)
            for i in range(len(CLL)):
                df.loc[len(df)]= [seed, dataName, size, 'TM', i, 'CLL', CLL[i]]
            for i in range(len(CLL),numIter):
                df.loc[len(df)]= [seed, dataName, size, 'TM', i, 'CLL', CLL[-1]]


            CLL= h.minibatchTM(X= X, Y= Y, size= 1, max_iter= numIter,seed=seed)
            for i in range(len(CLL)):
                df.loc[len(df)]= [seed, dataName, size, 'DEF', i, 'CLL', CLL[i]]
            for i in range(len(CLL),numIter):
                df.loc[len(df)]= [seed, dataName, size, 'DEF', i, 'CLL', CLL[-1]]

            mb_size= 5
            CLL= h.minibatchTM(X= X, Y= Y, size= mb_size, max_iter= numIter,seed=seed)
            for i in range(len(CLL)):
                df.loc[len(df)]= [seed, dataName, size, 'STM_'+str(mb_size), i, 'CLL', CLL[i]]
            for i in range(len(CLL),numIter):
                df.loc[len(df)]= [seed, dataName, size, 'STM_'+str(mb_size), i, 'CLL', CLL[-1]]

            mb_size= 10
            CLL= h.minibatchTM(X= X, Y= Y, size= mb_size, max_iter= int(numIter*size/mb_size),seed=seed)
            for i in range(len(CLL)):
                df.loc[len(df)]= [seed, dataName, size, 'STM_'+str(mb_size), i, 'CLL', CLL[i]]
            for i in range(len(CLL),numIter):
                df.loc[len(df)]= [seed, dataName, size, 'STM_'+str(mb_size), i, 'CLL', CLL[-1]]


    # Plot the results
    color = 'method'
    style= '$m$'
    num_colors = len(df[color].drop_duplicates().values)
    palette = sbn.color_palette("husl", num_colors)

    #for size in sizes:
    #    aux= df[df['$m$']== size]
    aux=df
    fig, ax = plt.subplots(1)
    sbn.lineplot(data=aux, x='n_iter', y='value', style=style, hue=color
                 , palette=palette).set_title(dataName)
    #                 , palette=palette, hue_order=['TM', 'STM', 'DEF']).set_title(dataName)
    #ax.set_xscale('log')
    plt.savefig(dataName + '.pdf', bbox_inches='tight')
    plt.show()


def evolucionCLL(structs= ["NB"], domain= "ParSum", ms=[500],ns=[10],seed= 1, r= 2):

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
    (CLL2,stats)=  h.learnCondMaxLikelihood(D[:,:-1],D[:,-1],esz=0.1,max_iter=10,trace=True)


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


from QDA import QDA
from LDA import LDA
from NaiveBayes import NaiveBayes
def ERD_models(dataNames= ['iris','glass',  'ionosphere', 'sonar','segment'], numIter= 15, num_rep=20):

    esz= 10.0
    for classifier in ['NB']:#,'LDA','QDA']:
        for dataName in dataNames:
            if classifier=='NB':
                # Discretize continuous variables
                D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1, bins=5)
                cardY = card[-1]
                card = card[:-1]
            else:
                D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1)
                cardY = card[-1]
            n = len(card) - 1
            m = D.shape[0]
            sizes = [int(m * 0.25), int(m * 0.5), int(m * 0.75)]
            size_test= int(m * 0.75)

            df = pandas.DataFrame(columns=['seed', 'data', 'size', 'n_iter', 'score', 'error'])
            for seed in range(num_rep):
                np.random.seed(seed)
                D = D[np.random.permutation(m), :]

                for size in sizes:
                    X = D[:size, :-1]
                    Y = D[:size, -1]
                    X_test = D[-size_test:, :-1]
                    Y_test = D[-size_test:, -1]

                    if classifier == 'LDA':
                        h0 = LDA(n, cardY)
                    elif classifier == 'QDA':
                        h0 = QDA(n, cardY)
                    elif classifier == 'NB':
                        h0 = NaiveBayes(n,cardY,card)
                    h0.fit(X,Y,size=esz)
                    p0= h0.getClassProbs(X)
                    p0_test= h0.getClassProbs(X_test)

                    error= np.average(Y!= np.argmax(p0, axis=1))
                    df.loc[len(df)] = [seed, dataName, size, 0, 'train', error]
                    error_test= np.average(Y_test!= np.argmax(p0_test, axis=1))
                    df.loc[len(df)] = [seed, dataName, size, 0, 'test', error_test]
                    min_error= error
                    df.loc[len(df)] = [seed, dataName, size, 0, 'train_min', min_error]
                    min_error_test= error_test
                    df.loc[len(df)] = [seed, dataName, size, 0, 'test_min', min_error_test]
                    #TODO poner la clasificacion en test

                    hi= list()
                    pY = p0.copy()
                    pY_test = p0_test.copy()
                    for iter in range(1,numIter):
                        if classifier == 'LDA':
                            hi.append(LDA(n, cardY))
                        elif classifier == 'QDA':
                            hi.append(QDA(n, cardY))
                        elif classifier == 'NB':
                            hi.append(NaiveBayes(n, cardY, card))
                        hi[-1].fit(X,pY,size= esz)

                        pY += p0 - hi[-1].getClassProbs(X)
                        #linear scaling and normalization: avoid probs >1 and <0
                        down = np.min(np.column_stack([np.min(pY, axis=1), np.zeros(size)]), axis=1)
                        up = np.max(np.column_stack([np.max(pY,axis= 1),np.ones(size)]),axis= 1)
                        pY = (pY - np.repeat(down,cardY).reshape((size,cardY)))/np.repeat(up- down,cardY).reshape((size,cardY))

                        pY_test += p0_test - hi[-1].getClassProbs(X_test)
                        #linear scaling and normalization: avoid probs >1 and <0
                        down = np.min(np.column_stack([np.min(pY_test, axis=1), np.zeros(size_test)]), axis=1)
                        up = np.max(np.column_stack([np.max(pY_test,axis= 1),np.ones(size_test)]),axis= 1)
                        pY_test = (pY_test - np.repeat(down,cardY).reshape((size_test,cardY)))/np.repeat(up- down,cardY).reshape((size_test,cardY))

                        error = np.average(Y != np.argmax(pY, axis=1))
                        df.loc[len(df)] = [seed, dataName, size, iter, 'train', error]
                        error_test = np.average(Y_test != np.argmax(pY_test, axis=1))
                        df.loc[len(df)] = [seed, dataName, size, iter, 'test', error_test]
                        if error < min_error:
                            min_error = error
                            min_error_test = error_test
                        df.loc[len(df)] = [seed, dataName, size, iter, 'train_min', min_error]
                        df.loc[len(df)] = [seed, dataName, size, iter, 'test_min', min_error_test]

                        if min_error < error:
                            print("data= "+dataName+" iter= "+str(iter)+" size= "+str(size))

            # Plot the results
            color = 'size'
            style = 'score'
            num_colors = len(df[color].drop_duplicates().values)
            palette = sbn.color_palette("husl", num_colors)

            # for size in sizes:
            #    aux= df[df['$m$']== size]
            aux = df
            fig, ax = plt.subplots(1)
            sbn.lineplot(data=aux, x='n_iter', y='error', style=style, hue=color, palette=palette).set_title(dataName+"_"+classifier)
            #                 , palette=palette, hue_order=['TM', 'STM', 'DEF']).set_title(dataName)
            # ax.set_xscale('log')
            plt.savefig("ERD_"+classifier+"_"+dataName + '.pdf', bbox_inches='tight')
            #plt.show()

def ERD_stats(dataNames= ['iris','glass',  'ionosphere', 'sonar','segment'], numIter= 15, num_rep=20):

    esz= 10.0
    for classifier in ['NB','LDA','QDA']:
        for dataName in dataNames:
            if classifier=='NB':
                # Discretize continuous variables
                D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1, bins=5)
                cardY = card[-1]
                card = card[:-1]
            else:
                D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1)
                cardY = card[-1]
            n = len(card) - 1
            m = D.shape[0]
            sizes = [int(m * 0.25), int(m * 0.5), int(m * 0.75)]
            size_test= int(m * 0.75)

            df = pandas.DataFrame(columns=['seed', 'data', 'size', 'n_iter', 'score', 'error'])
            for seed in range(num_rep):
                np.random.seed(seed)
                D = D[np.random.permutation(m), :]

                for size in sizes:
                    X = D[:size, :-1]
                    Y = D[:size, -1]
                    X_test = D[-size_test:, :-1]
                    Y_test = D[-size_test:, -1]

                    if classifier == 'LDA':
                        h0 = LDA(n, cardY)
                    elif classifier == 'QDA':
                        h0 = QDA(n, cardY)
                    elif classifier == 'NB':
                        h0 = NaiveBayes(n,cardY,card)
                    h0.fit(X,Y,size=esz)
                    p0= h0.getClassProbs(X)
                    p0_test= h0.getClassProbs(X_test)

                    error= np.average(Y!= np.argmax(p0, axis=1))
                    df.loc[len(df)] = [seed, dataName, size, 0, 'train', error]
                    error_test= np.average(Y_test!= np.argmax(p0_test, axis=1))
                    df.loc[len(df)] = [seed, dataName, size, 0, 'test', error_test]
                    min_error= error
                    df.loc[len(df)] = [seed, dataName, size, 0, 'train_min', min_error]
                    min_error_test= error_test
                    df.loc[len(df)] = [seed, dataName, size, 0, 'test_min', min_error_test]
                    #TODO poner la clasificacion en test

                    if classifier == 'LDA':
                        h = LDA(n, cardY)
                    elif classifier == 'QDA':
                        h = QDA(n, cardY)
                    elif classifier == 'NB':
                        h = NaiveBayes(n, cardY, card)
                    h.fit(X, Y, size=esz*5)

                    pY = p0.copy()
                    pY_test = p0_test.copy()
                    for iter in range(1, numIter):
                        if classifier == 'LDA':
                            hi = LDA(n, cardY)
                        elif classifier == 'QDA':
                            hi = QDA(n, cardY)
                        elif classifier == 'NB':
                            hi = NaiveBayes(n, cardY, card)
                        hi.fit(X, pY, size= esz)

                        # ERD updating rule
                        h.stats.add(h0.stats)
                        h.stats.subtract(hi.stats)
                        h.computeParams()

                        pY = h.getClassProbs(X)
                        #linear scaling and normalization: avoid probs >1 and <0
                        down = np.min(np.column_stack([np.min(pY, axis=1), np.zeros(size)]), axis=1)
                        up = np.max(np.column_stack([np.max(pY,axis= 1),np.ones(size)]),axis= 1)
                        pY = (pY - np.repeat(down,cardY).reshape((size,cardY)))/np.repeat(up- down,cardY).reshape((size,cardY))

                        pY_test = h.getClassProbs(X_test)
                        #linear scaling and normalization: avoid probs >1 and <0
                        down = np.min(np.column_stack([np.min(pY_test, axis=1), np.zeros(size_test)]), axis=1)
                        up = np.max(np.column_stack([np.max(pY_test,axis= 1),np.ones(size_test)]),axis= 1)
                        pY_test = (pY_test - np.repeat(down,cardY).reshape((size_test,cardY)))/np.repeat(up- down,cardY).reshape((size_test,cardY))

                        error = np.average(Y != np.argmax(pY, axis=1))
                        df.loc[len(df)] = [seed, dataName, size, iter, 'train', error]
                        error_test = np.average(Y_test != np.argmax(pY_test, axis=1))
                        df.loc[len(df)] = [seed, dataName, size, iter, 'test', error_test]
                        if error < min_error:
                            min_error = error
                            min_error_test = error_test
                        df.loc[len(df)] = [seed, dataName, size, iter, 'train_min', min_error]
                        df.loc[len(df)] = [seed, dataName, size, iter, 'test_min', min_error_test]

                        if min_error < error:
                            print("data= "+dataName+" iter= "+str(iter)+" size= "+str(size))

            # Plot the results
            color = 'size'
            style = 'score'
            num_colors = len(df[color].drop_duplicates().values)
            palette = sbn.color_palette("husl", num_colors)

            # for size in sizes:
            #    aux= df[df['$m$']== size]
            aux = df
            fig, ax = plt.subplots(1)
            sbn.lineplot(data=aux, x='n_iter', y='error', style=style, hue=color, palette=palette).set_title(dataName+"_"+classifier)
            #                 , palette=palette, hue_order=['TM', 'STM', 'DEF']).set_title(dataName)
            # ax.set_xscale('log')
            plt.savefig("ERD_stats_"+classifier+"_"+dataName + '.pdf', bbox_inches='tight')
            #plt.show()

from LogisticReg import LogisticReg
from CondMoments import  CondMoments
from GaussianNaiveBayes import GaussianNaiveBayes

alldatasets= ["iris", "sonar", "kr-vs-kp", "ionosphere", "breast-cancer", "balance-scale", "vowel", "letter"]

#LAS BUENAS: "vowel", "sonar", "letter", "kr-vs-kp", "breast-cancer", "balance-scale", "ionosphere",
#OK: "anneal", "glass","heart-statlog", "lymphography","primary-tumor", "segment", "soybean-small",

#ERROR: "audiology","autos","cleveland","hepatitis","horse-colic","labor","soybean-large","zoo.csv", "mushroom"

def det_prob(pY):

    m,r = pY.shape
    p_det= np.zeros((m,r))
    p_det[np.arange(m), np.argmax(pY, axis=1)] = 1

    return p_det

def grad_nb_params(X,Y,pY,p_y,mu_y,s2_y):
    '''
    gradient of the log loss wrt mu_y, sigma^2_y and p(y) for the Gaussian naive Bayes, loss= -1/m sum_{x,y} log p(y|x)
    :param X: unlabeled instances, numpy float
    :param Y: labels, numpy int
    :param pY: class probabilities given by the model, h(y|x)
    :param p_y: marginal probability distribution of the class. Model parameter
    :param mu_y: mean conditioned to the class labels. Model parameter
    :param s2_y: standard deviations conditioned to the class labels. Model parameter
    :return: (d loss/d p(y), d loss/d mu_y, d loss/ d sigma^2_y)
    '''
    m,n= X.shape
    r= pY.shape[1]
    e_y= np.zeros((m,r))
    e_y[np.arange(m),Y]= 1
    dmu_y = np.zeros((r,n))
    ds2_y = np.zeros((r,n))
    dp_y = np.zeros(r)
    for i in range(m):
        # d/d mu_z = -1/m sum_{x,y} (mu_z - x)/sigma^2_z * (delta(z=y) - p(y|x))
        dmu_y-= ((mu_y-X[i,:])/s2_y) * np.repeat(e_y[i,:]-pY[i,:],n).reshape((r,n))
        # d/d sigma^2_z = 1/m sum_{x,y} (x-mu_y)^2/(sigma^2_z^2 * (delta(z=y) - p(y|x))
        ds2_y+= ((X[i,:]-mu_y)**2 / (2 * s2_y**2)) * np.repeat(e_y[i,:]-pY[i,:],n).reshape((r,n))
        # d/d mu_z = -1/m sum_{x,y} 1/p(z) * (delta(z=y) - p(y|x))
        dp_y-= (1/p_y) * (e_y[i,:]-pY[i,:])

    return (dp_y/m, dmu_y/m, ds2_y/m)

def grad_nb_stats(X,Y,pY,p_y,m_y,s_y):
    '''
    gradient of the log loss wrt the moments order-1 and order-2 for the Gaussian naive Bayes, loss= -1/m sum_{x,y} log p(y|x)
    :param X: unlabeled instances, numpy float
    :param Y: labels, numpy int
    :param pY: class probabilities given by the model, h(y|x)
    :param p_y: marginal probability distribution of the class. Model parameter
    :param m_y: moment order-1 -average- conditioned to the class labels. statistics
    :param s_y: moment order-2 -average- conditioned to the class labels. Model parameter
    :return: (d loss/d m_y, d loss/ d s_y)
    '''
    m,n= X.shape
    r= pY.shape[1]
    e_y= np.zeros((m,r))
    e_y[np.arange(m),Y]= 1
    dm_y = np.zeros((r,n))
    ds_y = np.zeros((r,n))
    dp_y = np.zeros(r)
    for i in range(m):
        # d/d mu_z = -1/m sum_{x,y} (mu_z - x)/sigma^2_z * (delta(z=y) - p(y|x))
        dm_y = (m_y * X[i,:]**2 + (m_y**2 + s_y) * X[i,:] - m_y * s_y) / (m_y**4 + s_y**2 - 2 * m_y**2 * s_y) * np.repeat(e_y[i,:]-pY[i,:],n).reshape((r,n))
        # d/d sigma^2_z = 1/m sum_{x,y} (x-mu_y)^2/(sigma^2_z^2 * (delta(z=y) - p(y|x))
        ds_y = (X[i,:] - m_y)**2 / (2 * (m_y**2 + s_y)**2) * np.repeat(e_y[i,:]-pY[i,:],n).reshape((r,n))
        # d/d mu_z = -1/m sum_{x,y} 1/p(z) * (delta(z=y) - p(y|x))
        #dp_y-= (1/p_y) * (e_y[i,:]-pY[i,:])

    return (dm_y, ds_y)




def ERD_vs_gradient(dataNames= alldatasets, numIter= 30, num_rep=5):
    '''
    Test ERD aginst gradient descent with logistic regression using the equivalence between Gaussian naive Bayes and
    logistic regression under homocedasticity constraint, and ERD for Gaussian naive Bayes without homocedasticity.

    Esquema del gradient descent:
    -Initialization:
    maximum likelihood under the equivalence with naive Bayes under homocedasticity
    -Updating rules:
    1) beta_i_y<- beta_i_y + delta* sum_j=(1,m) x_i^j * (1(y=y^j) - h(y^j|x^j; beta,alpha)
    2) alpha_y<- alpha_y + delta* sum_j=(1,m) (1(y=y^j) - h(y^j|x^j; beta,alpha)

    :param dataNames: data sets
    :param numIter: numero of iterations for the iterative algorithms
    :param num_rep: number of repetitions of the experiment to account for the variability of the results
    :return:
    '''


    for dataName in dataNames:
        esz= 1.0

        df = pandas.DataFrame(columns=['seed', 'data', 'model', 'size', 'iter.', 'score', 'error'])
        for classifier in ['NB','LR']:
            for alg in ['Gradient', 'ERD', 'ERD det']:#'Gradient-NB',

                # Remove options not implemented
                if not (classifier == 'NB' or alg!= 'Gradient'):# and alg == 'Gradient'):

                    # step size for Gradient descent -log loss-: it should be of the order of Mv2^2 - see relation with initializ.
                    delta= 0.1
                    reg= 0.0

                    X, Y, card, cardY = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1)
                    n = len(card)
                    m = X.shape[0]

                    #Consider up to 750 instances for training and 250 for test
                    m= min([1000,m])
                    sizes = [int(m * 0.25), int(m * 0.5), int(m * 0.75)]
                    size_test= int(m * 0.75)
                    for seed in range(num_rep):
                        np.random.seed(seed)
                        perm= np.random.permutation(m)
                        X = X[perm,:]
                        Y = Y[perm]
                        for size in sizes:

                            print("data= " + dataName + " model= " + classifier + '+' + alg + ' seed=' + str(seed) + " size= " + str(size))
                            X_train = X[:size, :]
                            Y_train = Y[:size]
                            X_test = X[-size_test:, :]
                            Y_test = Y[-size_test:]

                            if classifier == 'LR':
                                h0 = LogisticReg(n, cardY)
                            elif classifier == 'NB':
                                h0 = GaussianNaiveBayes(n, cardY)

                            h0.fit(X_train,Y_train,size=esz)
                            '''
                            error_test= np.average(Y_test!= np.argmax(p0_test, axis=1))
                            df.loc[len(df)] = [seed, dataName, classifier, alg, size, 0, 'test', error_test]
                            min_error= error
                            df.loc[len(df)] = [seed, dataName, classifier, alg, size, 0, 'train_min', min_error]
                            min_error_test= error_test
                            df.loc[len(df)] = [seed, dataName, classifier, alg, size, 0, 'test_min', min_error_test]
                            '''

                            if classifier == 'LR':
                                h = LogisticReg(n, cardY)
                            elif classifier == 'NB':
                                h = GaussianNaiveBayes(n, cardY)

                            #h.fit(X_train, Y_train, size=esz)
                            h.init()
                            pY = h.getClassProbs(X_train)
                            pY= h.getClassProbs(X_train)

                            df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, 0, '0-1 loss', np.average(Y_train!= np.argmax(pY, axis=1))]
                            df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, 0, 'log loss', np.average([- np.log(pY[i,Y_train[i]]) for i in range(size)])]
                            df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, 0, 'rand 0-1 loss', np.average([1- pY[i,Y_train[i]] for i in range(size)])]

                            for iter in range(1, numIter):

                                if classifier == 'LR':
                                    hi = LogisticReg(n, cardY)
                                elif classifier == 'NB':
                                    hi = GaussianNaiveBayes(n, cardY)

                                try:
                                    if not alg=='ERD det':
                                        hi.fit(X_train, pY, size= esz)
                                    else:
                                        p_det= det_prob(pY)
                                        hi.fit(X_train, p_det, size= esz)
                                except:
                                    print("Fitting error: " + classifier + '+' + alg + " with size " + str(size) + " in " + str(iter))
                                    for i in range(iter,numIter):
                                        df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, i, '0-1 loss',
                                                           np.average(Y_train != np.argmax(pY, axis=1))]
                                        df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, i, 'log loss',
                                                           np.average([- np.log(pY[i, Y_train[i]]) for i in range(size)])]
                                        df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, i, 'rand 0-1 loss',
                                                           np.average([1 - pY[i, Y_train[i]] for i in range(size)])]
                                        break


                                # updating rules

                                if alg == 'Gradient' and classifier == 'LR':
                                    # beta_i_y <- beta_i_y + delta * 1/m sum_(x,y') x_i * (1(y=y') - h(y|x) - delta*reg*beta_i_y
                                    #h.beta_y += delta * (h0.stats.M1y[()] - hi.stats.M1y[()])/size - delta* reg* h.beta_y
                                    h.beta_y += delta * (h0.stats.M1y[()] - hi.stats.M1y[()]) / size - delta * reg * h.beta_y

                                    # alpha_y <- alpha_y + delta * sum_(x,y) (1(y=y ^ j) - h(y ^ j | x ^ j) - delta * reg * alpha_y
                                    h.alpha_y += delta * (h0.stats.M0y[()] - hi.stats.M0y[()]) / size - delta * reg * h.alpha_y

                                elif alg == 'Gradient' and classifier == 'NB':
                                    s_y = h.cov_y + h.mu_y**2
                                    dm_y, ds_y = grad_nb_stats(X_train, Y_train, pY, h.py, h.mu_y, s_y)

                                    h.mu_y -= delta * dm_y
                                    s_y -= delta * ds_y
                                    h.cov = s_y - h.mu_y**2

                                elif alg == 'ERD':
                                    h.stats.add(h0.stats, prop= delta)
                                    h.stats.subtract(hi.stats, prop= delta)
                                    h.computeParams()
                                elif alg == 'ERD det':
                                    h.stats.add(h0.stats, prop= delta)
                                    h.stats.subtract(hi.stats, prop= delta)
                                    h.computeParams()

                                pY = h.getClassProbs(X_train)
                                #linear scaling and normalization: avoid probs >1 and <0
                                down = np.min(np.column_stack([np.min(pY, axis=1), np.zeros(size)]), axis=1)
                                up = np.max(np.column_stack([np.max(pY,axis= 1),np.ones(size)]),axis= 1)
                                pY = (pY - np.repeat(down,cardY).reshape((size,cardY)))/np.repeat(up- down,cardY).reshape((size,cardY))

                                df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, iter, '0-1 loss',
                                                   np.average(Y_train != np.argmax(pY, axis=1))]
                                df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, iter, 'log loss',
                                                   np.average([- np.log(pY[i, Y_train[i]]) for i in range(size)])]
                                df.loc[len(df)] = [seed, dataName, classifier + '+' + alg, size, iter, 'rand 0-1 loss',
                                                   np.average([1 - pY[i, Y_train[i]] for i in range(size)])]
                                '''
                                error_test = np.average(Y_test != np.argmax(pY_test, axis=1))
                                df.loc[len(df)] = [seed, dataName, classifier, alg, size, iter, 'test', error_test]
                                if error < min_error:
                                    min_error = error
                                    min_error_test = error_test
                                df.loc[len(df)] = [seed, dataName, classifier, alg, size, iter, 'train_min', min_error]
                                df.loc[len(df)] = [seed, dataName, classifier, alg, size, iter, 'test_min', min_error_test]
        
                                if min_error < error:
                                    print("data= "+dataName+" iter= "+str(iter)+" size= "+str(size))
                                '''

        # Plot the results
        color = 'model'
        style = 'size'
        num_colors = len(df[color].drop_duplicates().values)
        palette = sbn.color_palette("husl", num_colors)


        for score in ["0-1 loss"]:#, "log loss", "rand 0-1 loss"]:
            aux= df[df['score']== score]
            fig, ax = plt.subplots(1)
            sbn.lineplot(data=aux, x='iter.', y='error', style=style, hue=color, palette=palette).set_title(dataName+" "+score  )
            #                 , palette=palette, hue_order=['TM', 'STM', 'DEF']).set_title(dataName)
            #ax.set_yscale('log')
            plt.savefig("./Results/ERD_vs_Grad_"+dataName+"_"+score+'.pdf', bbox_inches='tight')
            #plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ERD_vs_gradient()
    #pruebasCondMoments()

