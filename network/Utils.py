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


def generate_communication_sequence(size, values, prob, prob_stability_q):
    """
    Generates a random communication sequence
    :param size: Size of the output sequence
    :param values: Values to be used in the sequence
    :param prob: Probability of replacement in a subsequence
    :prob_prob_stability_q: Probability of repetition a subsequence
    """
    L = np.array([])
    s = _sample_sequence(values.copy(), prob)
    while len(L) < size:
        L = np.concatenate([L, s])
        if np.random.rand() > prob_stability_q:
            s = _sample_sequence(values.copy(), prob)
    return L[:size].astype(int).tolist()


def _sample_sequence(num_nodes, num_times):
    return np.concatenate([[i for i in range(num_nodes)] for j in range(num_times)])[np.random.permutation(num_nodes* num_times)]
