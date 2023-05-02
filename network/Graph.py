import Stats
from IBC import IBC
from network.UserSingleList import User
import numpy as np


class Graph:
    com_queue = []
    global_time = 1

    def __init__(self, adjacency_matrix, policy, data, structure, exec_sequence, split_type='uniform', data_domain=None):
        self.seed = 10
        np.random.seed(self.seed)
        self.policy = policy
        self.train_data = data
        card_x = np.array([len(np.unique(data[:, i])) for i in range(data[:, :-1].shape[1])])
        card_y = len(np.unique(data[:, -1]))
        data_split = self.split_data(data, len(adjacency_matrix), split_type=split_type)
        # Distribute data uniformly over the nodes
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
                               card_x=card_x,
                               card_y=card_y,
                               data_domain=data_domain if data_domain else None,
                               identifier=i)
                          for i in range(len(adjacency_matrix))]
        self.fill_children()
        self.adjacency_matrix = adjacency_matrix  # Rows => FROM, Columns => TO
        self.com_queue = exec_sequence

    def start(self, test_data):
        """
        Initial function to start the iterative algorithm
        :return: None
        """
        records = []
        stats_g = self.user_list[0].classifier.stats.emptyCopy()
        stats_g.uniform(self.train_data.shape[0])
        # stats_g.maximumLikelihood(self.train_data[:, :-1], self.train_data[:, -1], esz=0.1)
        classif_mle = self.user_list[0].classifier.copy()
        classif_mle.learnMaxLikelihood(self.train_data[:, :-1], self.train_data[:, -1], esz=0.1)
        while len(self.com_queue) > 0:
            # print(f"Global Time: {self.global_time}")
            # Get the next user from the queue
            # actual_user_ix = self.com_queue.pop()
            actual_user_ix = self.com_queue[0]
            del self.com_queue[0]
            actual_user = self.user_list[actual_user_ix]
            if len(actual_user.stats) == 0:
                actual_user.stats = [stats_g]
            # if self.global_time == 1:
            #     actual_user.stats = [stats_g]
            CLL = actual_user.compute(self.global_time, self.policy, self.train_data, test_data)
            CLL_mle = classif_mle.CLL(self.train_data[:, :-1], self.train_data[:, -1])
            for score in CLL.keys():
                to_insert = [self.seed, actual_user.data.shape[0], test_data.shape[0], actual_user.data.shape[0],
                             len(self.user_list), score, CLL[score], self.global_time, actual_user_ix, 1, self.policy]
                records.append(to_insert)

            records.append([self.seed, self.train_data.shape[0], self.train_data.shape[0], self.train_data.shape[0],
                             len(self.user_list), 'ML', CLL_mle, self.global_time, '-1', 1, self.policy])

            self.global_time += 1  # Increment the time when the communications are made
        return records

    def start_exact_TM_check(self, nodes_check, data, skip_first=True):
        """
        Checks if the Collaborative Learning environment matches an Exact TM with all data in a single user, without
        distribution nor collaborative settings
        """
        classif_structure = self.user_list[0].classifier_structure
        card_x = self.user_list[0].card_x
        card_y = self.user_list[0].card_y
        n_vars = self.user_list[0].n_vars

        h = IBC(card_x, card_y)
        if classif_structure == "NB":
            h.setBNstruct([[n_vars] for i in range(n_vars)] + [[]])
        elif classif_structure == "TAN":
            # (0,1), (1,2), ..., (n-2,n-1)
            h.setBNstruct([[n_vars]] + [[i - 1, n_vars] for i in range(1, n_vars)] + [[]])
        elif classif_structure == "2IBC":
            h.setKOrderStruct(k=2)

        print(f"Structure: {classif_structure}\tData shape: {data.shape}")
        while len(self.com_queue) > 0:
            # Get the next user from the queue
            actual_user_ix = self.com_queue.pop()
            actual_user = self.user_list[actual_user_ix]
            CLL = actual_user.compute(self.global_time, self.policy, data, data)  # We are not interested in CLL here!
            if actual_user_ix in nodes_check:
                if not skip_first:
                    preds_collaborative = actual_user.classifier.getClassProbs(data[:, :-1])

                    # TM in single user
                    u_t = None
                    h.learnCondMaxLikelihood(data[:, :-1], data[:, -1], stats=u_t, max_iter=1)
                    u_t = h.stats.copy()
                    preds_individual = h.getClassProbs(data[:, :-1])
                    diff = preds_individual - preds_collaborative
                    print(f"t={self.global_time})\t\tMSE-Diff: {np.nansum((diff)**2)}\t\tAbs-Diff: {np.nansum(diff)}")
                else:
                    skip_first = False
            self.global_time += 1  # Increment the time when the communications are made

    def fill_children(self):
        """
        Fills the children of every user in self.user_list with the pointers to their children for easier access.
        """
        for user in self.user_list:
            usr_children_lst = user.children
            usr_children_lst_full = [self.user_list[ch] for ch in usr_children_lst]
            user.children = usr_children_lst_full

    def split_data(self, data, n_splits, split_type):
        np.random.seed(self.seed)
        idx = np.arange(0, data.shape[0])  # Create the index column that is going to be split
        data_split = None
        if split_type == 'uniform':
            bins = np.linspace(0, n_splits, idx.size, endpoint=False).astype(int)
            bins = bins[idx.argsort().argsort()]
            data_split = [data[bins == i, :] for i in range(n_splits)]
        elif split_type == 'diritchlet':
            # TODO: Implement diritchlet
            raise "Not implemented yet"
        elif split_type == 'clustering':  # TODO Esto crea 1 cluster por cada nodo!!!
            from sklearn.cluster import KMeans
            mdl = KMeans(n_splits)
            mdl.fit(data[:, :-1])
            data_split = [data[mdl.labels_ == i, :] for i in np.unique(mdl.labels_)]
        return data_split
