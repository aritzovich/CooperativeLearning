from collections.abc import Iterable
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def show_graph(adjacency_matrix):
    """
    Plots the Directed Graph using the Networkx Graph library
    """

    G = nx.DiGraph(adjacency_matrix)
    nx.draw_networkx(G, node_size=2000)
    plt.show()


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


