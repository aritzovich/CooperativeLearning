import numpy as np
import argparse
from network.Graph import Graph


def main():
    np.random.seed(25)
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj_matrix', dest='adj_matrix', type=str,
                        help='Path to the adjacency matrix in TXT. Rows => FROM; Columns => TO')
    parser.add_argument('--max_iterations', dest='max_iterations', type=int,
                        help='Max iterations to run the experiment')
    parser.add_argument('--start_node', dest='start_node', type=int,
                        help='Index of the node to start with')
    parser.add_argument('--node_names', dest='node_names', type=str, nargs='+',
                        help='Node names')
    parser.add_argument('--policy', dest='policy', type=str,
                        help='Policy to be used "info" or "recent"')
    args = parser.parse_args()

    adj_matrix = np.loadtxt(args.adj_matrix)

    if args.node_names:
        node_names = args.node_names
    else:
        node_names = ['Node{}'.format(i) for i in range(0, len(adj_matrix))]
    start_node = node_names[args.start_node]  # To get the node_name of the start node

    # Now we call to the main function
    max_iterations = args.max_iterations
    print("The adjacency matrix that is going to be used is: ")
    print(adj_matrix)
    print("\n\n\n\n")
    graph = Graph(adj_matrix, max_iterations, start_node, node_names, args.policy, True)
    graph.start(start_node)


if __name__ == '__main__':
    main()
