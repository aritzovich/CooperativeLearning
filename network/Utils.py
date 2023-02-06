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
    :param xs: the list of lists to be flatten
    :return: the flatten list
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def plotParameters2D(line_id, x, y, score, savePath=None):
    """
    Makes a 2D scatterplot with a colorbar defined by score
    :param line_id: The ID of the different lines
    :param x: A list containing the x position of the points
    :param y: A list containing the y position of the points
    :param score: A list with the score obtained by each point
    """
    d = np.array([line_id, x, y, score]).T
    unique_types = np.unique(d[:, 0])
    for t in unique_types:
        subset = d[d[:, 0] == t, :]
        plt.plot(subset[:, 1], subset[:, 2], linestyle='-')
        plt.scatter(subset[:, 1], subset[:, 2], marker='o', c=subset[:, 3], s=200)
    plt.colorbar()
    if savePath:
        plt.savefig(savePath, format="pdf")
    else:
        plt.show()


def plotParameters3D(line_id, x, y, z, score, savePath=None):
    """
    Makes a 3D scatterplot with a colorbar defined by score
    :param line_id: The ID of the different lines
    :param x: A list containing the x position of the points
    :param y: A list containing the y position of the points
    :param z: A list containing the z position of the points
    :param score: A list with the score obtained by each point
    """
    d = np.array([line_id, x, y, z, score]).T

    ax = plt.figure().add_subplot(projection='3d')

    unique_types = np.unique(d[:, 0])
    for t in unique_types:
        subset = d[d[:, 0] == t, :]
        ax.plot(subset[:, 1], subset[:, 2], subset[:, 3], linestyle='-')
        p = ax.scatter(subset[:, 1], subset[:, 2], subset[:, 3], marker='o', c=subset[:, 4], s=200)
    plt.colorbar(p)
    if savePath:
        plt.savefig(savePath, format="pdf")
    else:
        plt.show()
