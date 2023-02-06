import numpy as np
import argparse

import pandas as pd

from GenerateData import generate_data_domains
from network.Graph import Graph


def main():
    np.random.seed(25)
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj_matrix', dest='adj_matrix', type=str,
                        help='Path to the adjacency matrix in TXT. Rows => FROM; Columns => TO')
    parser.add_argument('--exec_sequence', dest='exec_sequence', type=int, nargs='+',
                        help='Node names')
    parser.add_argument('--policy', dest='policy', type=str,
                        help='Policy to be used "info" or "recent"')
    parser.add_argument('--size', dest='size', type=int,
                        help='Length of the execution sequence to be generated randomly')
    parser.add_argument('--show', dest='show_network', type=int,
                        help='Plots the graph if True boolean specified')
    parser.add_argument('--data', dest='data', type=str,
                        help='Dataset to be used in the experiment. It will be uniformly distributed through nodes')
    parser.add_argument('--n_points', dest='n_points', type=int,
                        help='Number of points to be used to generate a dataset')
    parser.add_argument('--n_variables', dest='n_variables', type=int,
                        help='Number of variables that will be generated in the dataset')
    parser.add_argument('--dimension', dest='dimension', type=int,
                        help='Number of dimensions of each variable')
    parser.add_argument('--classif_structure', dest='structure', type=str,
                        help='Structure for the IBC classifier: "NB", "TAN", "2IBC"')
    parser.add_argument('--domain_problem', dest='p_domain', type=str,
                        help='Domain of the problem to generate the data accordingly')

    args = parser.parse_args()

    if not args.size and not args.exec_sequence:
        raise "You need to specify either an execution sequence or a size to generate a random one"

    if not args.data:
        data = generate_data_domains(domain=args.p_domain, m=args.n_points, n=args.n_variables+1, r=args.dimension)
    else:
        data = pd.read_csv(args.data, sep=',')
        data = data.values

    adj_matrix = np.loadtxt(args.adj_matrix)

    graph = Graph(adj_matrix, args.policy, data, args.structure,  data_domain=args.p_domain, exec_sequence=args.exec_sequence, show_graph=args.show_network)
    graph.start()


if __name__ == '__main__':
    main()
