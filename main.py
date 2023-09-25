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


import CondMoments as cmm
def pruebasCondMoments(dataName= "./data/iris.csv"):
     data,card= Utils.loadSupervisedData(dataName)
     X= data[:,:4]
     Y= np.array(data[:,4], dtype= np.int8)

     moments= cmm.CondMoments(n= 4,card=[],cardY=3,sets=[(4,),()])
     moments.maximumLikelihood(X,Y)



def pruebasAgregadoNB(dataNames= ['iris','glass','ionosphere','sonar','segment'], numIter= 10, num_rep=20):




    for dataName in dataNames:
        D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1, bins=3)
        n = len(card) - 1
        cardY = card[-1]
        card = card[:-1]
        m = D.shape[0]
        sizes = [int(m * 0.25), int(m * 0.5), int(m * 0.75)]
        X = D[:, :-1]
        Y = D[:, -1]

        df = pandas.DataFrame(columns=['seed', 'data', 'size', "n_iter", 'error'])
        for seed in range(num_rep):
            np.random.seed(seed)
            D = D[np.random.permutation(m), :]

            for size in sizes:
                X = D[:size, :-1]
                Y = D[:size, -1]

                nb = IBC.getNaiveBayesStruct(n)

                h0 = ibc.IBC(card, cardY)
                h0.setBNstruct(nb)
                h0.learnMaxLikelihood(X,Y)
                error= h0.error(X,Y)
                min_error= error
                df.loc[len(df)] = [seed, dataName, size, 0, error]

                p0= h0.getClassProbs(X)

                hi= list()
                pY= p0.copy()
                for iter in range(numIter):
                    hi.append(ibc.IBC(card, cardY))
                    hi[-1].setBNstruct(nb)
                    hi[-1].learnMaxWLikelihood(X,pY)

                    pY+= p0 - hi[-1].getClassProbs(X)
                    #linear scaling and normalization: avoid probs >1 and <0
                    down= np.min(np.column_stack([np.min(pY, axis=1), np.zeros(size)]), axis=1)
                    up= np.max(np.column_stack([np.max(pY,axis= 1),np.ones(size)]),axis= 1)
                    pYnorm= (pY - np.repeat(down,cardY).reshape((size,cardY)))/np.repeat(up- down,cardY).reshape((size,cardY))
                    predY= np.argmax(pYnorm,axis=1)

                    error= np.average(Y!=predY)

                    if min_error< error:
                        print("data= "+dataName+" iter= "+str(iter)+" size= "+str(size))

                    min_error= np.min([error,min_error])
                    df.loc[len(df)]= [seed, dataName, size, iter, min_error]

        # Plot the results
        color = 'size'
        style = 'size'
        num_colors = len(df[color].drop_duplicates().values)
        palette = sbn.color_palette("husl", num_colors)

        # for size in sizes:
        #    aux= df[df['$m$']== size]
        aux = df
        fig, ax = plt.subplots(1)
        sbn.lineplot(data=aux, x='n_iter', y='error', style=style, hue=color, palette=palette).set_title(dataName)
        #                 , palette=palette, hue_order=['TM', 'STM', 'DEF']).set_title(dataName)
        # ax.set_xscale('log')
        plt.savefig("ERD_"+dataName + '.pdf', bbox_inches='tight')
        #plt.show()


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
            X = D[:, :-1]
            Y = D[:, -1]
            X_test = D[-size_test:, :-1]
            Y_test = D[-size_test:, -1]

            df = pandas.DataFrame(columns=['seed', 'data', 'size', 'n_iter', 'score', 'error'])
            for seed in range(num_rep):
                np.random.seed(seed)
                D = D[np.random.permutation(m), :]

                for size in sizes:
                    X = D[:size, :-1]
                    Y = D[:size, -1]

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
            X = D[:, :-1]
            Y = D[:, -1]
            X_test = D[-size_test:, :-1]
            Y_test = D[-size_test:, -1]

            df = pandas.DataFrame(columns=['seed', 'data', 'size', 'n_iter', 'score', 'error'])
            for seed in range(num_rep):
                np.random.seed(seed)
                D = D[np.random.permutation(m), :]

                for size in sizes:
                    X = D[:size, :-1]
                    Y = D[:size, -1]

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

                        pY = h.getClassProbs(X_test)
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ERD_stats()
    #pruebasCondMoments()

