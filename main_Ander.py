import numpy as np
import argparse

import pandas as pd

from GenerateData import generate_data_domains, random_adj_generator, binarize_dataset, generate_topology, \
    generate_exec_sequence
from Utils import loadSupervisedData, plot_results
from network.Graph import Graph


def main():
    np.random.seed(10)
    parser = argparse.ArgumentParser(prog="Cooperative Learning - Bayesian Classifiers using the TM algorithm",
                                     description="This is the supporting code for the paper EMPTY. "
                                                 "Please consider citing the authors as: EMPTY.")

    g1 = parser.add_argument_group("Adjacency matrix", "Either introduce an adjacency matrix in a txt file or create "
                                                       "one randomly by providing '--n_users'.")
    g1.add_argument('--adj_matrix', dest='adj_matrix', type=str,
                    help='Path to the adjacency matrix in TXT. Rows => FROM; Columns => TO.')
    g1.add_argument('--n_users', dest='n_users', type=int,
                    help='Number of users to create an adjacency matrix.')
    g1.add_argument('--max_children', dest='max_children', type=int,
                    help="Maximum number of children for each user.")
    g1.add_argument('--topology', dest='topology', type=str,
                    help='Network topology to be generated: star, line or circle')

    g2 = parser.add_argument_group("Execution sequence", "Options related to the execution sequence of the "
                                                         "created nodes. You can supply an exec_sequence manually or "
                                                         "randomly generate one specifying its "
                                                         "length '--exec_seq_size'. The policy specifies how the "
                                                         "aggregations are made when receiving multiple statistics from"
                                                         "the different users.")
    g2.add_argument('--exec_sequence', dest='exec_sequence', type=int, nargs='+',
                    help='Sequence of node IDs.')
    g2.add_argument('--exec_seq_size', dest='exec_seq_size', type=int,
                    help='Length of the execution sequence to be generated randomly.')
    g2.add_argument('--exec_sequence_type', dest='exec_type', type=str,
                    help="Type of exec sequence to be generated. Either 'uniform' or 'perm_comb'")
    g2.add_argument('--policy', dest='policy', type=str,
                    help='Policy to be used "info" or "recent".')

    g3 = parser.add_argument_group("Data", "Arguments related to Data.")
    g3.add_argument('--train_data', dest='train_data', type=str,
                    help='Train data to be used in the experiment. It will be uniformly distributed through nodes.')
    g3.add_argument('--test_data', dest='test_data', type=str,
                    help='Test data to be used in the experiment. It will be uniformly distributed through nodes.')
    g3.add_argument('--n_points', dest='n_points', type=int,
                    help='Number of points to be used to generate a dataset.')
    g3.add_argument('--n_variables', dest='n_variables', type=int,
                    help='Number of variables that will be generated in the dataset.')
    g3.add_argument('--cardinality', dest='cardinality', type=int,
                    help='Number of dimensions of each variable.')
    g3.add_argument('--classifier_structure', dest='structure', type=str,
                    help='Structure for the IBC classifier: "NB", "TAN", "2IBC".')
    g3.add_argument('--domain_problem', dest='p_domain', type=str,
                    help='Domain of the problem to generate the data accordingly.')

    g4 = parser.add_argument_group("Plotting extra arguments", "Plotting options using networkx and matplotlib "
                                                               "python libraries.")
    g4.add_argument('--show', dest='show_network', type=int,
                    help='Plots the graph if "1"  specified.')

    args = parser.parse_args()

    if not args.exec_seq_size and not args.exec_sequence:
        raise "You need to specify either an execution sequence or a size to generate a random one"

    if not args.train_data or not args.test_data:
        train_data = generate_data_domains(domain=args.p_domain, m=args.n_points, n=args.n_variables+1, r=args.cardinality)
        test_data = generate_data_domains(domain=args.p_domain, m=args.n_points, n=args.n_variables+1, r=args.cardinality)
        data_name = 'generated'
    else:
        train_data, _ = loadSupervisedData(args.train_data, sep=',', skipHeader=0, classInd=None, maxDiscVals=5, bins=2)
        test_data, _ = loadSupervisedData(args.train_data, sep=',', skipHeader=0, classInd=None, maxDiscVals=5, bins=2)
        data_name = args.train_data

    if args.adj_matrix:
        adj_matrix = np.loadtxt(args.adj_matrix)
        topology = 'Given'
    else:
        if args.n_users and args.max_children and args.topology:
            adj_matrix = generate_topology(args.topology, args.n_users)
            topology = args.topology
        else:
            raise "You must specify '--n_users' and '--max_children' or provide an adjacency matrix with '--adj_matrix'"

    if not args.exec_sequence:
        if args.exec_type and args.exec_seq_size:
            exec_sequence = generate_exec_sequence(args.exec_type, args.exec_seq_size, len(adj_matrix))
            type_sequence = args.exec_type
        else:
            raise "Execution sequence not provided and unable to generate one due to missing arguments."
    else:
        exec_sequence = args.exec_sequence
        type_sequence = 'Given'

    if args.show_network:
        from network import Utils
        Utils.show_graph(adj_matrix)

    graph = Graph(adj_matrix, args.policy, train_data, args.structure, exec_sequence, data_domain=args.p_domain)
    results = graph.start(test_data)
    results = pd.DataFrame(results, columns=['seed', 'n_train', 'n_test', 'n_local_train', 'n_users',
                                             'score_name', 'score', 'time', 'id_user', 'n_local_iterations', 'policy'])
    results['data_name'] = data_name
    results['network_topology'] = topology
    results['type_sequence'] = type_sequence
    results['BN_Structure'] = args.structure

    print(results)

    results.to_csv('./results.csv', sep=',', decimal='.', index=False)
    print("The results have been exported to: ./results.csv")

    plot_results(results)



if __name__ == '__main__':
    main()
