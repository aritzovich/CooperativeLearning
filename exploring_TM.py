# This is a sample Python script.
import time

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

import IBC
import Stats
import Stats as st
import IBC as ibc
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sbn
import Utils as utl
import itertools as itr
from sklearn.manifold import MDS

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

def miniBatch(dataName= "iris", numBatches= 2, percTrain= 3/4, maxIter=50, seed= 0, numRep= 9, esz= 1, resFile= "results_miniBatch.csv"):
    '''
    Explore the minibatch TM.
    :return:
    '''

    #try:
    #    res= pd.read_pickle(resFile)
    #except:
    res= None

    D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1, bins=3)
    X= D[:,:-1]
    Y= D[:,-1]
    card= np.max(X,axis=0)+1
    cardY= np.max(Y)+1
    m, n= X.shape

    for seed in range(seed, seed+numRep):
        # Training test split
        np.random.seed(seed)
        perm= np.random.permutation(m)
        m_train= int(percTrain*m)
        trainX= X[perm[:m_train],:]
        trainY= Y[perm[:m_train]]
        testX= X[perm[m_train:],:]
        testY= Y[perm[m_train:]]

        # Create the batches from training
        mb= IBC.getMinibatchInds(m_train, int(m_train/numBatches))
        mbX= [trainX[mb[i],:] for i in range(len(mb))]
        mbY= [trainY[mb[i]] for i in range(len(mb))]

        # The classifier
        nb_struct= IBC.getNaiveBayesStruct(n)
        h= IBC.IBC(card,cardY)
        h.setBNstruct(nb_struct)


        # global TM
        globalCLL, Stats= h.learnCondMaxLikelihood(trainX,trainY,max_iter=maxIter,esz= esz, trace=True)
        testCLL= [h.CLL(testX,testY, stats= stats) for stats in Stats]
        mbCLL= [[h.CLL(mbX[i],mbY[i], stats= stats) for stats in Stats] for i in range(len(mb))]
        res= res_minibatch(seed, dataName, esz, m, n,'global', Stats, globalCLL, testCLL, mbCLL, res)

        # local TM for each batch
        mbStats= list()
        for ind_mb in range(len(mb)):
            Stats = h.learnCondMaxLikelihood(mbX[ind_mb], mbY[ind_mb], max_iter=maxIter, esz= esz/numBatches, trace=True)[1]
            mbStats.append(Stats[-1])
            testCLL = [h.CLL(testX, testY, stats=stats) for stats in Stats]
            globalCLL = [h.CLL(trainX,trainY, stats= stats) for stats in Stats]
            mbCLL = [[h.CLL(mbX[i], mbY[i], stats=stats) for stats in Stats] for i in range(len(mb))]

            res= res_minibatch(seed, dataName, esz, m, n, 'mb_'+str(ind_mb), Stats, globalCLL, testCLL, mbCLL, res)


        # minibatch TM
        Stats= h.minibatchTM(trainX, trainY, size= int(m_train/numBatches), max_iter=maxIter,trace=True, esz= esz, seed= seed, fixed_mb=True)[1]
        testCLL = [h.CLL(testX, testY, stats=stats) for stats in Stats]
        globalCLL = [h.CLL(trainX, trainY, stats=stats) for stats in Stats]
        mbCLL = [[h.CLL(mbX[i], mbY[i], stats=stats) for stats in Stats] for i in range(len(mb))]

        res = res_minibatch(seed, dataName, esz, m, n, 'minibatch', Stats, globalCLL, testCLL, mbCLL, res)

        Stats= [mbStats[0]]
        for i in range(1,len(mb)):
            Stats[0].add(mbStats[i])

        testCLL = [h.CLL(testX, testY, stats=stats) for stats in Stats]
        globalCLL = [h.CLL(trainX, trainY, stats=stats) for stats in Stats]
        mbCLL = [[h.CLL(mbX[i], mbY[i], stats=stats) for stats in Stats] for i in range(len(mb))]

        res = res_minibatch(seed, dataName, esz, m, n, 'avg.MB', Stats, globalCLL, testCLL, mbCLL, res)

    res.to_pickle(resFile)

def res_minibatch(seed, data, esz, m, n, method, stats, globalCLL, testCLL, mbCLL, res= None):

    if res is None:
        res = pd.DataFrame(columns=['seed', 'data', 'esz', '$m$', '$n$', 'method', 'n_iter', 'stats', 'score', 'value'])

    #U= [(i,) for i in range(n)]

    for i in range(len(globalCLL)):
        ser= stats[i].serialize()
        res.loc[len(res)] = [seed, data, esz, m, n, method, i, ser, 'global', globalCLL[i]]
        res.loc[len(res)] = [seed, data, esz, m, n, method, i, ser, 'generalization', testCLL[i]]
        for j in range(len(mbCLL)):
            res.loc[len(res)] = [seed, data,esz, m, n, method, i, ser, 'mb_' + str(j), mbCLL[j][i]]

    return res

def plot_minibatch(resFile= "results_miniBatch.csv"):

    res = pd.read_pickle(resFile)
    data = 'iris'
    color = 'method'
    style = 'score'
    num_colors = len(res[color].drop_duplicates().values)
    palette = sbn.color_palette("husl", num_colors)


    aux = res.loc[res['score'] == 'global', :]
    #aux = aux.loc[res['seed'] == 0, :]
    mds = MDS(n_components=2)
    embedded = mds.fit_transform([stats[:-1]/stats[-1] for stats in pd.DataFrame(aux['stats'].to_list()).values])
    aux['$N_0$']=embedded[:,0]
    aux['$N_1$']=embedded[:,1]
    color = 'method'
    style = 'score'

        
    g = sbn.FacetGrid(aux, col="seed", hue=color, palette=palette, col_wrap=3)
    g.map(sbn.scatterplot, '$N_0$', '$N_1$')
    g.add_legend()
    g.savefig(data + '_scatterplot-ml.pdf', bbox_inches='tight')

    aux = res.loc[res['score'] == 'global', :]
    # aux = aux.loc[res['seed'] == 0, :]
    mds = MDS(n_components=2)
    embedded = mds.fit_transform([stats[:-1] / stats[-1] for stats in pd.DataFrame(aux['stats'].to_list()).values])
    aux['$N_0$'] = embedded[:, 0]
    aux['$N_1$'] = embedded[:, 1]
    color = 'method'
    style = 'score'

    sbn.lineplot()



    '''

    'seed', 'data', 'esz', '$m$', '$n$', 'method', 'n_iter', 'stats', 'score', 'value'

    aux = res
    # Plot the lines on two facets
    g= sbn.lineplot(data=aux, x="n_iter", y="value", hue="method", size="score", kind="line",
        size_order= ['global','generalization'], palette=palette
    )# Plot the lines on two facets
    g.add_legend()
    g.savefig(data + '_lineplot.pdf', bbox_inches='tight')

    #fig, ax = plt.subplots(np.unique(aux['seed']))
    #sbn.scatterplot(data=aux.loc[aux['seed'] == seeds.__next__(), :], x='$N_0$', y='$N_1$', style=style, hue=color, palette=palette).set_title(data)
    #plt.savefig(data + '.pdf', bbox_inches='tight')
    #plt.show()

    #acceder a las filas de pandas dataframe: iloc (con indices) y loc con pandas series y con listas
    #res.loc[res['method']== 'global',:]
    '''

def initialization_effect(dataName= "iris", max_iter= 50, num_rep= 10, seed= 0):
    '''
    Analizar el efecto de la inicializacion en minibatch y en TM. 
    
    Inicializaciones:
    -Uniforme: uniforme 
    -Informativa: emplear parte de los datos diferentes del training
    -Maximo verosimil: emplear el training
    
    Equivalent sample size de la inicializacion:
    -Tamaño del training
    -10 veces el training    
    
    Metodos:
        Stochastic: update con cada instancia
        Minibatch: Dos batches
        TM: update con todas las instancias de training
        
    
    
    :param data: data set
    :param esz: equivalent size of the maximum likelihood initial statistics
    :param b: number of batches of the minicath
    :param max_iter: maximum number of iterations
    :param numRep: number of repetitions of the experiment 
    :param seed: initial random seed, (seed,...,seed+num_rep-1)
    :return: saves a data frame with the obtained results
    '''

    #try:
    #    res= pd.read_pickle(resFile)
    #except:
    res= None

    D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1, bins=3)
    X= D[:,:-1]
    Y= D[:,-1]
    card= np.max(X,axis=0)+1
    cardY= np.max(Y)+1
    m, n= X.shape

    percTrain= 0.66

    for seed in range(seed, seed+num_rep):
        # Training test split
        np.random.seed(seed)
        perm= np.random.permutation(m)
        m_train= int(percTrain*m)
        trainX= X[perm[:m_train],:]
        trainY= Y[perm[:m_train]]
        testX= X[perm[m_train:],:]
        testY= Y[perm[m_train:]]


        # The classifier
        nb_struct= IBC.getNaiveBayesStruct(n)
        h= IBC.IBC(card,cardY)
        h.setBNstruct(nb_struct)
        h.initStats()

        #TODO meter el equivalent sample size
        '''
        Equivalent sample size:
        -Tamaño del training
        -10 veces el training    
        '''
        for esz in [int(m_train/2.0), m_train, m_train*2, m_train*10]:

            '''
            Inicializaciones:
            -Uniforme: uniforme 
            -Informativa: emplear parte de los datos diferentes del training
            -Maximo verosimil: emplear el training
            '''

            #TODO testar
            initializ= ["Uniform","Validation","Training",'Tra.+Unif.']
            Stats0= list()
            # Uniforme
            stats= h.stats.emptyCopy()
            stats.initCounts(esz=esz)
            Stats0.append(stats)
            # Informativa
            stats = h.stats.emptyCopy()
            stats.maximumLikelihood(X= testX, Y=testY, esz= 1)
            stats.setSampleSize(esz)
            Stats0.append((stats))
            # Maximo verosimil
            stats = h.stats.emptyCopy()
            stats.maximumLikelihood(X= trainX, Y=trainY, esz= 1)
            stats.setSampleSize(esz)
            Stats0.append((stats))
            # Maximo verosimil
            stats = h.stats.emptyCopy()
            stats.maximumLikelihood(X= trainX, Y=trainY, esz= m_train)
            stats.setSampleSize(esz)
            Stats0.append((stats))


            for indInit in range(len(initializ)):
                initStats = Stats0[indInit]

                '''
                Metodos:
                Stochastic: batch de tamaño 1
                Minibatch: 5 y 2 batches
                TM: todo el training
                '''
                method= ["stochastic","minibatch_10","minibatch_5","minibatch_2","TM"]
                for indMethod in range(len(method)):


                    if indMethod== 0:
                        CLL= h.learnMinLogLoss(trainX,trainY,mb_size=1, init_stats=initStats, max_iter= max_iter, seed= seed)
                    elif indMethod== 1:
                        CLL= h.learnMinLogLoss(trainX,trainY,mb_size=int(m_train/10), init_stats=initStats, max_iter= max_iter, seed= seed)
                    elif indMethod== 2:
                        CLL= h.learnMinLogLoss(trainX,trainY,mb_size=int(m_train/5), init_stats=initStats, max_iter= max_iter, seed= seed)
                    elif indMethod== 3:
                        CLL= h.learnMinLogLoss(trainX,trainY,mb_size=int(m_train/2), init_stats=initStats, max_iter= max_iter, seed= seed)
                    else:
                        CLL= h.learnMinLogLoss(trainX,trainY,init_stats=initStats, max_iter= max_iter, seed= seed)

                    #if len(CLL)< max_iter: CLL= np.concatenate([CLL,np.ones(max_iter-len(CLL))*CLL[-1]])

                    res= res_initialization_effect(esz= esz, initializ= initializ[indInit], method= method[indMethod], CLL= CLL, data_name= dataName, m= m_train, n= n, seed= seed, res= res)


    res.to_csv("results_initialization_"+dataName+".csv")


def res_initialization_effect(esz, initializ, method, CLL, data_name, m, n, seed, res= None):
    '''
    Generate the data frame for the experiment "initialization_effect"

    :param esz: Equivalent sample size of the initial statistics
    :param initializ: Initialization type
    :param method: Learning procedure
    :param CLL: Evolution of the conditional likelihood
    :param data_name: name of the data set
    :param m: size of the data
    :param n: number of variables
    :param seed: random seed of the experiments
    :param res: previous results
    :return:
    '''
    if res is None:
        res = pd.DataFrame(columns=['esz', 'init.', 'method', 'n_iter', 'CLL', 'data_name', 'm', 'n', 'seed'])

    for i in range(len(CLL)):
        res.loc[len(res)] = [esz, initializ, method, i, CLL[i], data_name, m, n, seed]

    return res


def plot_initialization_effect(dataName= "iris"):
    resFile= "results_initialization_"+dataName+".csv"
    res = pd.read_csv(resFile, index_col=0)


    # Plot the responses for different events and regions
    sbn.set_theme(style="darkgrid")

    values= res['esz'].drop_duplicates().values
    for esz in values:
        aux = res.loc[res['esz'] == esz, :]

        # Plot the responses for different events and regions
        lineplot= sbn.lineplot(x="n_iter", y="CLL",
                     hue="method", style="init.",
                     data=aux)
        lineplot.set_title(dataName+"; esz= "+str(esz))
        plt.xscale('log')
        plt.ylim(-0.4,-0.1)
        plt.savefig("initialization_"+dataName + '_'+str(esz)+'.pdf', bbox_inches='tight')
        #plt.show()
        plt.close()



if __name__ == '__main__':
    initialization_effect(max_iter= 50, num_rep= 20)
    plot_initialization_effect()
