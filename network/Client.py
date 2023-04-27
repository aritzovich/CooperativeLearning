import numpy as np

from Stats import Stats
from GenerateData import generate_data_domains
import IBC

class Client:

    def __init__(self, identifier, classif_structure, data, card_x, card_y, data_domain):
        self.id = identifier
        self.stats_old = [{}]  # This list contains, dicts of the following form NodeID: (Stats, #time)
        self.stats_new = [{}]  # This list contains, dicts of the following form NodeID: (Stats, #time)
        self.data = data
        self.data_domain = data_domain
        self.n_vars = self.data.shape[1]
        self.card_x = card_x
        self.card_y = card_y
        self.classifier = IBC.IBC(self.card_x, self.card_y)
        self.classifier_structure = classif_structure
        m, n = data.shape
        if classif_structure == "NB":
            nb = IBC.getNaiveBayesStruct(n - 1)
            self.classifier.setBNstruct(nb)
        elif classif_structure == "TAN":
            # (0,1), (1,2), ..., (n-2,n-1)
            self.classifier.setBNstruct([[self.n_vars]] + [[i - 1, self.n_vars] for i in range(1, self.n_vars)] + [[]])
        elif classif_structure == "2IBC":
            self.classifier.setKOrderStruct(k=2)
        else:
            raise "ERROR::Provide a Valid Structure: 'NB', 'TAN', '2IBC' "
        self.stats_ref = Stats(self.n_vars, self.card_x, self.card_y)
        self.stats_ref.maximumLikelihood(self.data[:, :-1], self.data[:, -1])

    def expectation(self, global_time, expectation_statistics, lr=1, esz=0.1):
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
            u_g = expectation_statistics[0][0].copy()
            for k in range(1, len(expectation_statistics)):
                u_g.add(expectation_statistics[k][0])
            # Compute the u(X,u_g) with the global
            u_t = self.classifier.stats.copy()  # Save u_t first so we can update it later
            self.classifier.setStats(u_g)  # Update the current classifier with u_g
            probs = self.classifier.getClassProbs(self.data[:, :-1])
            u_t.update(self.data[:, :-1], probs, self.stats_ref, lr=lr, esz=esz)
            result = {self.id: (u_t, global_time)}
        else:
            # The received list is empty so maximum likelihood parameters
            self.classifier.learnMaxLikelihood(self.data[:, :-1], self.data[:, -1], esz=esz)
            result = {self.id: (self.classifier.stats, global_time)}
        error = self.classifier.error(self.data[:, :-1], self.data[:, -1])
        return result

    # def evaluate_classifier(self, data_test=None, n_points=1000):
    #     """
    #     :param data_test: numpy matrix with data. If not supplied data is generated
    #     :param n_points: number of data points to be generated.
    #     """
    #     if not data_test:
    #         data_test = generate_data_domains(self.data_domain, n_points, self.data.shape[1], self.card_x[0])
    #     error = self.classifier.error(data_test[:, :-1], data_test[:, -1])
    #     print(f"n= {n_points} m= {self.data.shape[1]} dis-{self.classifier_structure}+LL:\t test: {error}")
    #     return error
