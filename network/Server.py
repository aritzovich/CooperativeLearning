import numpy as np

from network.Statistic import Statistic


class Server:

    def __init__(self, identifier):
        self.id = identifier
        self.theta2 = []
        self.theta1 = []

    def aggregate(self, thetaA, thetaB):
        """
        Aggregates the sufficient statistics
        :param thetaA: A list of statistics to aggregate
        :param thetaB: Another list of statistics to aggregate
        :return: the aggregated list of statistics
        """
        agg_stats = None
        if len(thetaA) == 0:
            agg_stats = thetaB
        elif len(thetaB) == 0:
            agg_stats = thetaA
        else:
            agg_stats = thetaA + thetaB
            agg_stats = self.clean_duplicates(agg_stats)
        return agg_stats

    def clean_duplicates(self, agg_stats):
        new_agg_stats = agg_stats.copy()
        to_delete = []
        for ix_s_r, s_r in enumerate(agg_stats):
            for ix_s in range(ix_s_r+1, len(agg_stats)):
                s = agg_stats[ix_s]
                if s.name == s_r.name:
                    if s.time > s_r.time:
                        to_delete.append(ix_s_r)
                    else:
                        to_delete.append(ix_s)
        if len(to_delete) > 0:
            to_delete = np.array(to_delete)
            new_agg_stats = np.array(new_agg_stats)
            new_agg_stats = np.delete(new_agg_stats, to_delete).tolist()
        return new_agg_stats
