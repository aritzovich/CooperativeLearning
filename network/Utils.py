from collections.abc import Iterable
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def show_graph(adjacency_matrix, export_path=None):
    """
    Plots the Directed Graph using the Networkx Graph library
    """

    G = nx.DiGraph(adjacency_matrix)
    nx.draw_networkx(G, node_size=2000)
    if export_path:
        plt.savefig(export_path, format='pdf')
    else:
        plt.show()
    plt.clf()


def flatten(xs):
    """
    A generic function to flatten a list of lists
    :param xs: the list of lists to be flattened
    :return: the flattened list
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def generate_communication_sequence(size, num_nodes, num_times, prob_stability_q):
    """
    Generates a random communication sequence
    :param size: Size of the output sequence
    :param num_nodes: num_nodes to be used in the sequence
    :param num_times: Number of times in which a node is repeated in a subsequence
    :prob_prob_stability_q: Probability of repetition a subsequence
    """
    L = np.array([])
    s = _sample_sequence(num_nodes, num_times)
    while len(L) < size:
        L = np.concatenate([L, s])
        if np.random.rand() > prob_stability_q:
            s = _sample_sequence(num_nodes, num_times)
    return L[:size].astype(int).tolist()


def _sample_sequence(num_nodes, num_times):
    return np.concatenate([[i for i in range(num_nodes)] for j in range(num_times)])[np.random.permutation(num_nodes* num_times)]

def kruskal(self, n, E=None, W=None, seed= None):
    '''
    Kruskal's algorithm (for maximization)
    Complexity O(|E|log n)

    E: candidate edges, E[i] represent an edge (u,v). If None generate all the candidate sets
    W: weights associated to the candidate edges, W[i] corresponds to E[i]. If None sample random weights

    return a list of edges representing a forest, list((u,v))
    '''

    if seed is not None:
        np.random.seed(seed)

    tree = list()
    treeW = list()
    V = [i for i in range(n)]
    ds = DisjointSet(n)
    dictV = dict()
    for i in range(n):
        dictV.update({V[i]: i})

    if W is None:
        if E is None:
            E = [(i,j) for i in range(n-1) for j in range(i+1,n)]

        W= np.random.random(len(E))

    sort = np.argsort(-np.array(W))
    ind = 0
    while len(tree) < len(V) - 1 and ind < len(E):
        e = E[sort[ind]]
        if ds.find(dictV[e[0]]) != ds.find(dictV[e[1]]):
            tree.append(e)
            treeW.append(W[sort[ind]])
            ds.union(dictV[e[0]], dictV[e[1]])

        ind += 1

    return (tree, treeW)

class DisjointSet():
    '''
    Disjoint-set forest structure -see Wikipedia.

    Used in the efficient implementation of the Kruskal's Algorithm
    '''

    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int)

    def find(self, x):
        '''
        Finds the root of the tree and performs path compression, that is,
        it flattens the structure of the tree and creates a star structure

        O(n)
        '''
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        '''
        Union by rank:
        Union by rank always attaches the shorter tree to the root of the taller tree

        O(n)
        '''

        xRoot = self.find(x)
        yRoot = self.find(y)

        # x and y are already in the same set
        if xRoot == yRoot:
            return

        # x and y are not in same set, so we merge them
        if self.rank[xRoot] < self.rank[yRoot]:
            self.parent[xRoot] = yRoot
        elif self.rank[xRoot] > self.rank[yRoot]:
            self.parent[yRoot] = xRoot
        else:
            # Arbitrarily make one root the new parent
            self.parent[yRoot] = xRoot
            self.rank[xRoot] += 1