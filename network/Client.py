import numpy as np

from GenerateData import generate_data_domains
from IBC import IBC


class Client:

    def __init__(self, identifier, classif_structure, data, data_domain):
        self.id = identifier
        self.stats_old = [{}]  # This list contains, dicts of the following form NodeID: (Stats, #time)
        self.stats_new = [{}]  # This list contains, dicts of the following form NodeID: (Stats, #time)
        self.data = data
        self.data_domain = data_domain
        n_vars = self.data.shape[1]-1
        self.card_x = np.array([len(np.unique(v)) for v in data.T[:-1]])
        self.card_y = len(np.unique(self.data[:, -1]))
        self.classifier = IBC(self.card_x, self.card_y)
        self.classif_structure = classif_structure
        if classif_structure == "NB":
            self.classifier.setBNstruct([[n_vars] for i in range(n_vars)] + [[]])
        elif classif_structure == "TAN":
            # (0,1), (1,2), ..., (n-2,n-1)
            self.classifier.setBNstruct([[n_vars]] + [[i - 1, n_vars] for i in range(1, n_vars)] + [[]])
        elif classif_structure == "2IBC":
            self.classifier.setKOrderStruct(k=2)
        else:
            raise "ERROR::Provide a Valid Structure: 'NB', 'TAN', '2IBC' "

    def expectation(self, global_time, expectation_statistics, esz=0):
        """
        Learns the maximum likelihood statistics from data or from received statistics
        :param global_time: Global time to be used when generating the statistics
        :param expectation_statistics: List of received statistics from other users/clients
        :return: the resulting tuple after expectation (Stats, time)
        """
        result = None
        if expectation_statistics:
            # If the received list contains data
            probs = self.classifier.getClassProbs(self.data[:, :-1])
            for k in expectation_statistics:
                self.classifier.stats = self.classifier.stats.update(self.data[:, :-1], probs,
                                                                     expectation_statistics[k][0], lr=1.0, esz=0.1)
            result = {self.id: (self.classifier.stats, global_time)}
        else:
            # The received list is empty so maximum likelihood parameters
            self.classifier.learnMaxLikelihood(self.data[:, :-1], self.data[:, -1], esz=0.1)
            result = {self.id: (self.classifier.stats, global_time)}
        return result

    def evaluate_classifier(self, data_test=None, n_points=1000):
        """
        :param data_test: numpy matrix with data. If not supplied data is generated
        :param n_points: number of data points to be generated.
        """
        if not data_test:
            data_test = generate_data_domains(self.data_domain, n_points, self.data.shape[1], self.card_x[0])
        error = self.classifier.error(data_test[:, :-1], data_test[:, -1])
        print(f"n= {n_points} m= {self.data.shape[1]} dis-{self.classif_structure}+LL:\t test: {error}")
        return error
