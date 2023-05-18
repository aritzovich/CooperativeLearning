import numpy as np

import IBC
import Stats


class User:

    def __init__(self, children, parents, identifier, classif_structure, data, card_x, card_y, data_domain=None, esz=0.1):
        self.children = children
        self.parents = parents
        self.id = identifier
        self.stats = []  # This list contains, dicts of the following form NodeID: (Stats, #time)
        self.data = data
        self.data_domain = data_domain
        self.n_vars = self.data.shape[1]
        self.card_x = card_x
        self.card_y = card_y
        self.classifier = IBC.IBC(self.card_x, self.card_y)
        self.classifier_structure = classif_structure
        self.esz = esz
        self.initialized = False
        m, n = data.shape
        if classif_structure == "NB":
            nb = IBC.getNaiveBayesStruct(n - 1)
            self.classifier.setBNstruct(nb)
        elif classif_structure == "TAN":
            # (0,1), (1,2), ..., (n-2,n-1)
            self.classifier.setBNstruct(
                [[self.n_vars]] + [[i - 1, self.n_vars] for i in range(1, self.n_vars)] + [[]])
        elif classif_structure == "2IBC":
            self.classifier.setKOrderStruct(k=2)
        else:
            raise "ERROR::Provide a Valid Structure: 'NB', 'TAN', '2IBC' "
        self.stats_ref = self.classifier.stats.emptyCopy()
        # Stats.Stats(self.n_vars, self.card_x, self.card_y)
        self.stats_ref.maximumLikelihood(self.data[:, :-1], self.data[:, -1], esz=self.esz)

    def compute(self, global_time, policy, global_train_data, test_data):
        """
        Performs the corresponding actions in a user level.
        :param global_time: Global time to update the time of the computed statistics
        :param global_train_data: The entire Training data set that has been split among the users
        :param test_data: A Test dataset to compute generalization error
        :return: A dictionary containing CLL_local, CLL_global and CLL_test
        """

        # Do the average
        if len(self.stats) == 0:
            stats_averaged = self.classifier.stats
        else:
            stats_averaged = Stats.average(self.stats)

        result = self.expectation(global_time, expectation_statistics=stats_averaged)
        self.classifier.setStats(result)

        for ch in self.children:
            ch.enqueue_stats(result)
        self.stats = []

        # Compute the CLL
        results = {'CLL_local': self.classifier.CLL(self.data[:, :-1], self.data[:, -1], normalize=True),
                   'CLL_global': self.classifier.CLL(global_train_data[:, :-1], global_train_data[:, -1],
                                                     normalize=True),
                   'CLL_test': self.classifier.CLL(test_data[:, :-1], test_data[:, -1], normalize=True),
                   'logloss_local': self.classifier.error(self.data[:, :-1], self.data[:, -1],
                                                          deterministic=False),
                   'logloss_global': self.classifier.error(global_train_data[:, :-1], global_train_data[:, -1],
                                                           deterministic=False),
                   'logloss_test': self.classifier.error(test_data[:, :-1], test_data[:, -1],
                                                         deterministic=False)
                   }
        return results

    def expectation(self, global_time, expectation_statistics, lr=1):
        """
        Learns the maximum likelihood statistics from data or from received statistics
        :param global_time: Global time to be used when generating the statistics
        :param expectation_statistics: List of received statistics from other users/clients
        :param lr: Learning rate
        :param esz: Equivalent sample size
        :return: the resulting tuple after expectation (Stats, time)
        """
        result = None
        if expectation_statistics:
            # If the received list contains data
            # First get the u_g (the global stats)
            stats_g = expectation_statistics.copy()
            # Compute the stats(X,stats_g) with the global
            self.classifier.setStats(stats_g)  # Update the current classifier with stats_g
            probs = self.classifier.getClassProbs(self.data[:, :-1])
            # acc = 0
            # for i, c in enumerate(self.data[:, -1]):
            #     acc += 1 - probs[i, c]
            # acc /= self.data.shape[0]
            # print(f"Inercia: {acc}")

            stats_g.update(self.data[:, :-1], probs, self.stats_ref, lr=lr, esz=self.esz)
            stats_g.checkConsistency()
            result = stats_g
        else:
            print("NOOOOO")
            # The received list is empty so maximum likelihood parameters
            self.classifier.learnMaxLikelihood(self.data[:, :-1], self.data[:, -1], esz=self.esz)
            result = self.classifier.stats
        error = self.classifier.error(self.data[:, :-1], self.data[:, -1])
        return result

    def enqueue_stats(self, stats):
        """
        Enqueues the received statistics
        :param stats: Received statistics
        :return: None
        """
        self.initialized = True
        self.stats.append(stats)

