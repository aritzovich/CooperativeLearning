import numpy as np
import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from GenerateData import generate_data_domains, generate_topology, generate_exec_sequence
import IBC
from Utils import loadSupervisedData, plotParameters2D


def main():
    SEED = 10
    np.random.seed(SEED)
    parser = argparse.ArgumentParser(prog="Bayesian Classifiers using the TM algorithm - Mini Batches",
                                     description="This is the supporting code for the paper EMPTY. "
                                                 "Please consider citing the authors as: EMPTY.")

    parser.add_argument('--data', dest='data', type=str,
                        help='Path to the data that will be split into train, test.')

    parser.add_argument('--n_batches', dest='n_batches', type=int,
                        help='Number of mini batches that will be created from the training set')

    parser.add_argument('--classifier_structure', dest='structure', type=str,
                        help='Structure for the IBC classifier: "NB", "TAN", "2IBC".')

    parser.add_argument('--percentage_split', dest='p_split', type=int,
                        help='Percentage to use for train data. Between 0-100.')

    args = parser.parse_args()
    data, card = loadSupervisedData(args.data, sep=',', skipHeader=0, classInd=None, maxDiscVals=5, bins=2)

    n_rows = data.shape[0]
    rows_train = np.random.choice(range(n_rows), int(n_rows*args.p_split/100))
    train = data[rows_train, :]
    test = data[-rows_train, :]

    train_batches = []
    mini_batch_size = int(len(train)/args.n_batches)
    for mini_batch in range(args.n_batches):
        rows_mini_batch = np.random.choice(list(range(len(rows_train))), mini_batch_size, replace=True)
        train_batches.append(train[rows_mini_batch])

    # The data is created. Now move to learn individual IBC and run TM until convergence

    classifiers = []
    CLLs_independent = []
    ids_independent = []
    for i, d in enumerate(train_batches):
        m, n = d.shape
        h = IBC.IBC(card[:-1], card[-1])
        nb = IBC.getNaiveBayesStruct(n-1)
        h.setBNstruct(nb)
        CLL, stats = h.learnCondMaxLikelihood(d[:, :-1], d[:, -1], trace=True)
        ids_independent.append(np.repeat(i, len(CLL)))
        CLLs_independent.append(CLL)
        classifiers.append(h)

    # Now we extract all the classifier stats and perform a PCA so we can plot in 2D
    stats_lst_independent = [h.stats.serialize() for h in classifiers]
    pca = PCA(n_components=2)
    points2D_independet = pca.fit_transform(stats_lst_independent)
    # TODO Usar el MDS -> SKLEARN

    # Now we do the same but sharing and aggregating between the batches  -- MiniBatch!
    m, n = train.shape
    max_iter = 100
    diff = np.inf
    h = IBC.IBC(card[:-1], card[-1])
    nb = IBC.getNaiveBayesStruct(n - 1)
    h.setBNstruct(nb)
    h.learnMaxLikelihood(train[:, :-1], train[:, -1])
    max_likelihood_stats = h.stats
    stats_lst_minibatch = [max_likelihood_stats]
    # df_mb = pd.DataFrame({'iter_minibatch': [], 'score_name': [], 'score_values': [], 'stats': []})
    df_mb = []
    for i in range(max_iter):
        CLL_train, stats_train = h.minibatchTM(X=train[:, :-1], Y=train[:, -1], seed=SEED, size=int(m/args.n_batches),
                                                 trace=True, stats=stats_lst_minibatch[-1], max_iter=1)
        # Get the randOrdering of the minibatches
        np.random.seed(SEED)
        randOrder = np.random.permutation(m)
        size = int(m/args.n_batches)
        for ix_stats in range(len(stats_train)):
            # Obtain CLL for test
            CLL_test = h.CLL(test[:, :-1], test[:, -1], stats=stats_train[ix_stats])
            aux_test = [i, 'CLL_test', CLL_test, stats_train[ix_stats]]
            df_mb.append(aux_test)
            # df_mb = pd.concat([df_mb, pd.DataFrame(aux_test)], ignore_index=True)
            for ind_mb in range(args.n_batches):
                indexes = [randOrder[i] for i in range(ind_mb * size % m, (ind_mb + 1) * size % m if (ind_mb + 1) * size % m > ind_mb * size % m else m)] + [randOrder[i] for i in range(0 if (ind_mb + 1) * size % m > ind_mb * size % m else (ind_mb + 1) * size % m)]
                CLL_local = h.CLL(train[indexes, :-1], train[indexes, -1], stats=stats_train[ix_stats])
                # Store the data
                # aux_l = {'iter_minibatch': i, 'score_name': 'CLL_local_' + str(ind_mb),
                         # 'score_values': CLL_local, 'stats': stats_train[ind_mb]}
                aux_l = [i, 'CLL_local_' + str(ind_mb), CLL_local, stats_train[ind_mb]]
                df_mb.append(aux_l)

        # aux_g = {'iter_minibatch': i, 'score_name': 'CLL_global', 'score_values': CLL_train, 'stats': stats_train}
        aux_g = [i, 'CLL_global', CLL_train, stats_train]
        df_mb.append(aux_g)
        stats_lst_minibatch.append(stats_train)

    # aux_indep = {'iter_minibatch': -1, 'score_name': 'CLL_global',
    #              'score_values': CLLs_independent, 'stats': stats_lst_independent}
    aux_indep = [-1, 'CLL_global', CLLs_independent, stats_lst_independent]
    df_mb.append(aux_indep, ignore_index=True)
    embedding = MDS(n_components=2, normalized_stress='auto')
    df_mb['stats2D'] = embedding.fit_transform(df_mb['stats'])

    plotParameters2D(df_mb['iter_minibatch'], df_mb['stats2D'][:, 0], df_mb['stats2D'][:, 1],
                     df_mb.iloc[df_mb['score_name'] == 'CLL_global', :]['score_values'], savePath=None)




if __name__ == '__main__':
    main()
