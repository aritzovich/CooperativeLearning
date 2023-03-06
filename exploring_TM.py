# This is a sample Python script.
import time

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas
import sklearn as sk

import IBC
import Stats as st
import IBC as ibc
import ContStats as cst
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sbn
import Utils as utl

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

def miniBatch(dataName= "iris", numBatches= 3, percTrain= 1/3, maxIter= 10, seed= 0, numRep= 10):
    '''
    Explore the minibatch TM.
    :return:
    '''

    D, card = utl.loadSupervisedData(dataName='./data/' + dataName + '.csv', skipHeader=1, bins=3)
    X= D[:,:-1]
    Y= D[:,-1]
    card= np.max(X,axis=0)+1
    cardY= np.max(Y)+1
    n,m= X.shape

    for seed in range(seed, seed+numRep):
        # Training test split
        np.random.seed(seed)
        perm= np.random.permutation()
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
        globalCLL, Stats= h.learnCondMaxLikelihood(trainX,trainY,max_iter=maxIter,trace=True)
        testCLL= [h.CLL(testX,testY, stats= stats) for stats in Stats]
        mbCLL= [[h.CLL(mbX[i],mbY[i], stats= stats) for stats in Stats] for i in range(len(mb))]

        # local TM for each batch
        for ind_mb in range(len(mb)):
            Stats = h.learnCondMaxLikelihood(mbX[ind_mb], mbY[ind_mb], max_iter=maxIter, trace=True)[1]
            testCLL = [h.CLL(testX, testY, stats=stats) for stats in Stats]
            globalCLL = [h.CLL(trainX,trainY, stats= stats) for stats in Stats]
            mbCLL = [[h.CLL(mbX[i], mbY[i], stats=stats) for stats in Stats] for i in range(len(mb))]

        # minibatch TM
        Stats= h.minibatchTM(trainX,trainY,max_iter=maxIter,trace=True,seed= seed)[1]
        testCLL = [h.CLL(testX, testY, stats=stats) for stats in Stats]
        globalCLL = [h.CLL(trainX, trainY, stats=stats) for stats in Stats]
        mbCLL = [[h.CLL(mbX[i], mbY[i], stats=stats) for stats in Stats] for i in range(len(mb))]



if __name__ == '__main__':
    miniBatch()
