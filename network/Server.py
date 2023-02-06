import numpy as np

from network.Statistic import Statistic


class Server:

    def __init__(self, identifier):
        self.id = identifier

    def aggregate(self, thetaA, thetaB):
        """
        Aggregates the sufficient statistics
        :param thetaA: A list of statistics to aggregate
        :param thetaB: Another list of statistics to aggregate
        :return: the aggregated list of statistics
        """
        if len(thetaA) == 0:
            agg_stats = thetaB
        elif len(thetaB) == 0:
            agg_stats = thetaA
        else:
            agg_stats = thetaB.copy()
            for k in thetaA.keys():
                if k not in agg_stats.keys():
                    agg_stats[k] = thetaA[k]
                else:
                    if thetaA[k][1] > thetaB[k][1]:
                        agg_stats[k] = thetaA[k]
                    else:
                        agg_stats[k] = thetaB[k]
        return agg_stats

    # def clean_duplicates(self, agg_stats):
    #     # Aritz propone sets
    #     """
    #     Cleans any duplicates in the specified agg_stats list. In case of two duplicates, it takes the newest statistic
    #     :param agg_stats: a 1D list with the sufficient statistics
    #     :return: a list without duplicated statistics
    #     """
    #     new_agg_stats = agg_stats.copy()
    #     to_delete = []
    #     for ix_s_r, s_r in enumerate(agg_stats):
    #         for ix_s in range(ix_s_r+1, len(agg_stats)):
    #             s = agg_stats[ix_s]
    #             if s[1] == s_r[1]:
    #                 if s[2] > s_r[2]:
    #                     to_delete.append(ix_s_r)
    #                 else:
    #                     to_delete.append(ix_s)
    #     if len(to_delete) > 0:
    #         to_delete = np.array(to_delete)
    #         new_agg_stats = np.array(new_agg_stats)
    #         new_agg_stats = np.delete(new_agg_stats, to_delete).tolist()
    #     return new_agg_stats
