import numpy as np
import pandas as pd
from network.Statistic import Statistic


class Client:

    def __init__(self, identifier):
        self.id = identifier
        self.theta2 = []
        self.theta1 = []

    def expectation(self, global_time, expectation_statistics):
        """
        Learns the maximum likelihood statistics from data or from received statistics
        :param global_time: Global time to be used when generating the statistics if necessary
        :param expectation_statistics: List of received statistics from other users/clients
        :return: the resulting expectation
        """
        result = None
        if expectation_statistics:
            # If the received list contains data
            result = [Statistic(self.id, str(self.id), global_time)]
        else:
            # The received list is empty so maximum likelihood parameters
            result = [Statistic(self.id, str(self.id), global_time)]
        return result
