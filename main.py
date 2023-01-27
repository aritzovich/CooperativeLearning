# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import Stats as st
import IBC as ibc

def pruebas(structs= ["2IBC"], domain= "ParSum", ms=[1000],ns=[15],seed= 1, r= 2):

    np.random.seed(seed)
    mTest= 1000

    for n in ns:
        for m in ms:
            card = np.ones(n, dtype=int)* r
            cardY = r

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
                print("n= "+str(n)+" m= "+str(m)+" dis-"+struct + "+LL:\t train: "+ str(h.error(D[:,:-1], D[:,-1]))
                      + "\t test: "+ str(h.error(T[:,:-1], T[:,-1])))

                plt.plot(np.array(CLL)/(m*n),label="n="+str(n)+"\tm="+str(m))

    plt.legend(loc="upper left")
    plt.show()




def generateData(domain= "ParSum", m=10, n=5, r= 4):
    if(domain == "parity"):
        p_flip= 0.1
        D= np.random.choice(a=[0,1],size=(m,n))
        D[:,n-1]= (D[:,n-1]+np.random.choice(a=[0,1],size=m,p=[1-p_flip, p_flip]))%2

    elif(domain == "ParSum"):
        D = np.random.choice(a=[0,1], size=(m, n))
        fx= sum([(-1)**(D[:,2*i]+D[:,2*i+1]) for i in range(int(n/2))])
        py_x= 1/(1+ np.exp(-fx))
        for i in range(m):
            D[i,-1]= np.random.choice(a=[0,1], size= 1, p=[1-py_x[i], py_x[i]])[0]

    else:
        D = np.random.choice(a=[i for i in range(r)], size=(m, n))

    return D


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pruebas()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
