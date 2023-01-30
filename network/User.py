from network.Client import Client
from network.Server import Server


class User(Client, Server):

    def __init__(self, children, parents, identifier):
        super().__init__(identifier)
        super().__init__(identifier)
        self.children = children
        self.parents = parents

    def compute(self, global_time, policy):
        """
        Performs the corresponding actions in a user level.
        :param global_time: Global time to update the time of the computed statistics
        :return: theta2 and theta2
        """
        theta2, theta1 = None, None
        if len(self.theta1) > 1:
            # This node has received multiple statistics from multiple nodes, we need to resolve this before computing
            self.resolve_multiple_lists()

        inside_theta1, inside_theta2 = self.is_statistic_inside()
        if not inside_theta2:
            theta1 = self.theta1[0]
            theta2 = self.aggregate(self.theta2[0], self.expectation(global_time, []))
        elif inside_theta2 and not inside_theta1:
            theta1 = self.aggregate(self.theta1[0], self.expectation(global_time, self.theta2[0]))
            theta2 = self.theta2[0]
        elif inside_theta2 and inside_theta1:
            # Decide depending on policy
            if policy == "info":
                if len(self.theta1[0]) > len(self.theta2[0]):
                    theta2 = self.theta1[0]
                else:
                    theta2 = self.theta2[0]
            elif policy == "recent":
                creation_time_theta1 = [s.time for s in self.theta1[0] if s.id == self.id]
                creation_time_theta2 = [s.time for s in self.theta2[0] if s.id == self.id]
                if creation_time_theta1 > creation_time_theta2:
                    theta2 = self.theta1[0]
                else:
                    theta2 = self.theta2[0]
            theta1 = self.aggregate(self.theta1[0], self.expectation(global_time, theta2))
        else:
            print(f"Situation not considered (NODEID {self.id}):\n\tTheta2 = {self.theta2}\n\t Theta1 = {self.theta1}")

        for ch in self.children:
            ch.enqueue_stats(theta2, theta1)

        self.theta1 = []
        self.theta2 = []
        return theta2, theta1

    def is_statistic_inside(self):
        inside_theta1 = False
        inside_theta2 = False
        for s in self.theta1[0]:
            if s.name == self.id:
                inside_theta1 = True
        for s in self.theta2[0]:
            if s.name == self.id:
                inside_theta2 = True
        return inside_theta1, inside_theta2

    def enqueue_stats(self, theta2, theta1):
        """
        Enqueues the received statistics
        :param theta2: Old received statistics
        :param theta1: New received statistics
        :return: None
        """
        self.theta2.append(theta2)
        self.theta1.append(theta1)

    def resolve_multiple_lists(self):
        """
        Selects one theta1 and theta2 from the available statistics lists. It is possible that the resulting thetaX is
        an aggregation of multiple thetas but without mixing elements from theta1 and theta2.
        """
        # First select one theta1
        tA = self.theta1[0]
        for it in range(1, len(self.theta1)):
            tB = self.theta1[it]
            tA = self.aggregate(tA, tB)
        theta1 = tA
        # Now do the same with theta2
        tA = self.theta2[0]
        for it in range(1, len(self.theta2)):
            tB = self.theta2[it]
            tA = self.aggregate(tA, tB)
        theta2 = tA

        # # We get the one with more statistics
        # lengths = [len(l) for l in self.theta1]
        # theta1 = self.theta1[lengths.index(max(lengths))]
        # # Select one theta2
        # # We get the one with most statistics
        # lengths = [len(l) for l in self.theta2]
        #
        # theta2 = self.theta2[lengths.index(max(lengths))]

        self.theta1 = [theta1]
        self.theta2 = [theta2]
