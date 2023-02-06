from network.Client import Client
from network.Server import Server


class User(Client, Server): 

    def __init__(self, children, parents, identifier, classif_structure, data, data_domain=None):
        super(User, self).__init__(identifier, classif_structure, data, data_domain)  # For the Client
        self.children = children
        self.parents = parents

    def compute(self, global_time, policy):
        """
        Performs the corresponding actions in a user level.
        :param global_time: Global time to update the time of the computed statistics
        :param policy: The user defined policy. If policy = info, the expectation is always made maximizing information
        if policy = recent it computes the expectation with the most recent statistics
        :return: Two lists containing the old statistics (stats_old) and the newest statistics (stats_new). No statistics from
        stats_old have been used to compute statistics of stats_new.
        """
        stats_old, stats_new = None, None
        if len(self.stats_new) > 1:
            # This node has received multiple statistics from multiple nodes, we need to resolve this before computing
            self.resolve_multiple_lists()

        # inside_theta1, inside_theta2 = self.is_statistic_inside()
        inside_theta1 = self.id in self.stats_new[0].keys()
        inside_theta2 = self.id in self.stats_old[0].keys()
        if not inside_theta2:
            stats_new = self.stats_new[0]
            stats_old = self.aggregate(self.stats_old[0], self.expectation(global_time, []))
        elif inside_theta2 and not inside_theta1:
            stats_new = self.aggregate(self.stats_new[0], self.expectation(global_time, self.stats_old[0]))
            stats_old = self.stats_old[0]
        elif inside_theta2 and inside_theta1:
            # Decide depending on policy
            if policy == "info":
                if len(self.stats_new[0]) > len(self.stats_old[0]):
                    stats_old = self.stats_new[0]
                else:
                    stats_old = self.stats_old[0]
            elif policy == "recent":
                creation_time_theta1 = [s[2] for s in self.stats_new[0] if s[1] == self.id]
                creation_time_theta2 = [s[2] for s in self.stats_old[0] if s[1] == self.id]
                if creation_time_theta1 > creation_time_theta2:
                    stats_old = self.stats_new[0]
                else:
                    stats_old = self.stats_old[0]
            stats_new = self.aggregate(self.stats_new[0], self.expectation(global_time, stats_old))
            stats_old = self.stats_new[0]
        else:
            print(f"Situation not considered (NODEID {self.id}):\n\tTheta2 = {self.stats_old}\n\t Theta1 = {self.stats_new}")

        for ch in self.children:
            ch.enqueue_stats(stats_old, stats_new)

        self.stats_new = []
        self.stats_old = []
        return stats_old, stats_new

    # def is_statistic_inside(self):
    #     """
    #     Checks whether the user statistic is inside stats_old and stats_new
    #     :return: A tuple with two boolean values corresponding to is inside stats_new and is inside stats_old respectively
    #     """
    #     inside_theta1 = False
    #     inside_theta2 = False
    #     for s in self.stats_new[0]:
    #         if s[1] == self.id:
    #             inside_theta1 = True
    #     for s in self.stats_old[0]:
    #         if s[1] == self.id:
    #             inside_theta2 = True
    #     return inside_theta1, inside_theta2

    def enqueue_stats(self, stats_old, stats_new):
        """
        Enqueues the received statistics
        :param stats_old: Old received statistics
        :param stats_new: New received statistics
        :return: None
        """
        self.stats_old.append(stats_old)
        self.stats_new.append(stats_new)

    def resolve_multiple_lists(self):
        """
        Selects one stats_new and stats_old from the available statistics lists. It is possible that the resulting
        stats_lst is an aggregation of multiple stats_lst but without mixing elements from stats_new and stats_old.
        """
        # First select one stats_new
        tA = self.stats_new[0]
        for it in range(1, len(self.stats_new)):
            tB = self.stats_new[it]
            tA = self.aggregate(tA, tB)
        stats_new = tA
        # Now do the same with stats_old
        tA = self.stats_old[0]
        for it in range(1, len(self.stats_old)):
            tB = self.stats_old[it]
            tA = self.aggregate(tA, tB)
        stats_old = tA

        self.stats_new = [stats_new]
        self.stats_old = [stats_old]
