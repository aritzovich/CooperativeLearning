from collections.abc import Iterable

from network.User import User
from scipy.stats import norm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    com_queue = []
    global_time = 1

    def __init__(self, adjacency_matrix, max_iterations, start_node, node_names, policy, show_graph=True):
        self.policy = policy
        self.node_name_num_correspondence = {}
        for ix, nm in enumerate(node_names):
            self.node_name_num_correspondence[nm] = ix
        self.adjacency_matrix = adjacency_matrix  # Rows => FROM, Columns => TO
        self.max_iterations = max_iterations
        self.com_queue.insert(0, start_node)
        self.com_queue = list(flatten(self.com_queue))  # To flatten the list
        self.com_manager = adjacency_matrix - 1  # Creates the com manager

        self.G = nx.DiGraph()  # Create an empty directed graph
        # Now create the edges
        for ix_r, row in enumerate(self.adjacency_matrix):
            children_node = np.squeeze(np.where(self.adjacency_matrix[ix_r, :]))
            if np.size(children_node) > 1:
                children_node = [node_names[c] for c in children_node]  # So that we always index by node_name
            else:
                children_node = [node_names[children_node]]
            parents_node = np.squeeze(np.where(self.adjacency_matrix[:, ix_r]))
            if np.size(parents_node) > 1:
                parents_node = [node_names[c] for c in parents_node]
            else:
                parents_node = [node_names[parents_node]]
            self.G.add_node(node_names[ix_r], data=User(children_node, parents_node, node_names[ix_r]))
            for ix_c, column in enumerate(self.adjacency_matrix):
                if self.adjacency_matrix[ix_r, ix_c]:
                    self.G.add_edge(node_names[ix_r], node_names[ix_c])
        self.replace_children_ix_with_data()
        # # Set the ancestors for the Users
        # self.ancestor_control = np.zeros((len(node_names), len(node_names)))
        # self.ancestor_control.fill(-1)  # We set to -1
        # self.ancestor_control_template = np.copy(self.ancestor_control)  # We store the template for next iterations
        # for nm in node_names:
        #     nm_ix = self.node_name_num_correspondence[nm]
        #     ancestors = nx.ancestors(self.G, nm)
        #     for a in ancestors:
        #         a_ix = self.node_name_num_correspondence[a]
        #         self.ancestor_control[nm_ix, a_ix] = 0  # We initialize to False
        # Show the graph
        if show_graph:
            self.show_graph()

    def show_graph(self):
        """
        Plots the Directed Graph
        :return:
        """
        nx.draw_networkx(self.G, node_size=2000)
        plt.show()

    def start(self, node_name):
        """
        Initial function to start the iterative algorithm
        :param node_name: Node name to start from
        :return: None
        """
        self.initialize_com_queue()
        actual_user = self.get_user_data(self.com_queue[len(self.com_queue)-1])
        actual_user.enqueue_stats([], [])

        while self.max_iterations > 0 and len(self.com_queue) > 0:
            # Get the next user from the queue
            actual_user_name = self.com_queue.pop()
            actual_user = self.get_user_data(actual_user_name)

            print(f"===> GLOBAL TIME: {self.global_time}")
            print("Quién soy: " + actual_user_name)
            print("Qué tengo:")
            print(f"\tTheta 2: {actual_user.theta2}")
            print(f"\tTheta 1: {actual_user.theta1}")
            print(f"A quién mando: {[str(c.id)+', '  for c in actual_user.children]}")
            print(f"Qué mando:")
            theta2, theta1 = actual_user.compute(self.global_time, self.policy)
            print(f"\tTheta 2: {theta2}")
            print(f"\tTheta 1: {theta1}")
            print("")
            print("")
            self.global_time += 1  # Increment the time when the communications are made
        print("FIN")

    def initialize_com_queue(self):
        """
        Hard-coded version for testing. It will be removed in further commits
        """
        # l = ['Node0', 'Node1', 'Node0', 'Node1', 'Node2', 'Node0']
        l = ['Node0', 'Node3', 'Node0', 'Node3', 'Node1', 'Node2', 'Node0']
        l.reverse()
        self.com_queue = l

    def generate_random_com_queue(self, size):
        """
        Generates a random communication queue with the specified size
        """
        node_names = list(self.G.nodes.keys())
        l = np.random.choice(node_names, size, replace=True)
        self.com_queue = l.tolist()

    def get_user_data(self, node_name):
        """
        Returns the data of the specific networkx node
        """
        return self.G.nodes(data=True)[node_name]['data']

    def restore_ancestor_control(self):
        self.ancestor_control = np.copy(self.ancestor_control_template)

    def ancestors_completed(self):
        """
        Checks if there is a user who has received information from all of his ancestors. If True, then the user index
        is returned and the ancestors_control is restored with the previously stored template
        :return: Returns which user has received data from all the ancestors.
        If none has received data yet, returns False
        """
        for row_ix in range(len(self.ancestor_control)):
            row = self.ancestor_control[row_ix, :]
            row = row[row != -1]
            if np.all(row):
                self.restore_ancestor_control()
                return row_ix
        return False

    def replace_children_ix_with_data(self):
        """
        Replaces the children index list with the children data so it is easier to access to the children.
        """
        for node_name in self.G.nodes:
            node = self.get_user_data(node_name)
            children_list = []
            parents_list = []
            for ch_ix in node.children:
                ch_data = self.get_user_data(ch_ix)
                children_list.append(ch_data)
            for p_ix in node.parents:
                p_data = self.get_user_data(p_ix)
                parents_list.append(p_data)
            node.children = children_list
            node.parents = parents_list
            
def flatten(xs):
    """
    A generic function to flatten a list of lists
    :param xs: the list of lists to be flatten
    :return: the flatten list
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x