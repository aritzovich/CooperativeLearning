from network import Utils
from network.User import User
import numpy as np


class Graph:
    com_queue = []
    global_time = 1

    def __init__(self, adjacency_matrix, policy, data, structure, data_domain=None, exec_sequence=None, size=None, show_graph=False):
        self.policy = policy
        # Distribute data uniformly over the nodes
        # TODO: DIRITCHLET?
        idx = np.arange(0, data.shape[0])  # Create the index column that is going to be split
        n_splits = len(adjacency_matrix)
        bins = np.linspace(0, n_splits, idx.size, endpoint=False).astype(int)
        bins = bins[idx.argsort().argsort()]
        # for b in np.unique(bins):
        #     self.user_list[b].data = data[bins == b, :]
        self.user_list = [User(children=np.where(adjacency_matrix[i, :] == 1)[0].tolist(),
                               parents=np.where(adjacency_matrix[:, i] == 1)[0].tolist(),
                               classif_structure=structure,
                               data=data[bins == i, :],
                               data_domain=data_domain if data_domain else None,
                               identifier=i)
                          for i in range(len(adjacency_matrix))]
        self.fill_children()
        self.adjacency_matrix = adjacency_matrix  # Rows => FROM, Columns => TO
        if exec_sequence:
            self.com_queue = exec_sequence
        elif exec_sequence == "generate":
            self.generate_random_com_queue(size=size)

        if show_graph:
            Utils.show_graph(adjacency_matrix)

    def start(self):
        """
        Initial function to start the iterative algorithm
        :return: None
        """

        while len(self.com_queue) > 0:
            # Get the next user from the queue
            actual_user_ix = self.com_queue.pop()
            actual_user = self.user_list[actual_user_ix]

            print(f"===> GLOBAL TIME: {self.global_time}")
            print(f"Quién soy: {actual_user_ix}")
            print("Qué tengo:")
            print(f"\tStats Old: {actual_user.stats_old}")
            print(f"\tStats New: {actual_user.stats_new}")
            print(f"A quién mando: {[c.id for c in actual_user.children]}")
            print(f"Qué mando:")
            stats_old, stats_new = actual_user.compute(self.global_time, self.policy)
            print(f"\tStats Old: {['(' + str(s) + '-' + str(stats_old[s][1]) + ')' for s in stats_old]}")
            print(f"\tStats New: {['(' + str(s) + '-' + str(stats_new[s][1]) + ')' for s in stats_new]}")
            print("")
            print("")
            self.global_time += 1  # Increment the time when the communications are made
        print("FIN")

    def generate_random_com_queue(self, size):
        """
        Generates a random communication queue with the specified size
        """
        l = np.random.choice(self.user_list, size, replace=True)
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

    def fill_children(self):
        for user in self.user_list:
            usr_children_lst = user.children
            usr_children_lst_full = [self.user_list[ch] for ch in usr_children_lst]
            user.children = usr_children_lst_full
