import itertools
import numpy as np


def generate_data(n_points, dimension, n_variables, dist_prob="uniform"):
    joint_prob = create_joint_prob(n_variables, dimension, dist_prob)
    data = generate_data_based_on_prob(n_points, joint_prob, dimension, n_variables)
    return data, joint_prob


def generate_data_based_on_prob(num_samples, joint_probs, dimension, n_variables):
    possibilities = np.array(dimension ** n_variables)
    indexes = np.random.choice(possibilities, p=joint_probs[:, -1], size=num_samples)
    data = joint_probs[indexes, 0:n_variables]
    return data


def create_joint_prob(variables, dimension, distribution):
    lst = list(itertools.product(np.array(range(dimension)), repeat=variables))
    lst = np.array(lst)
    probs = None
    if distribution == "uniform":
        probs = np.random.rand(dimension ** variables)
    elif distribution == "norm":
        probs = np.abs(np.random.normal(size=dimension ** variables))
    joint_probs = np.zeros((lst.shape[0], lst.shape[1] + 1))
    joint_probs[:, :-1] = lst
    joint_probs[:, -1] = probs / sum(probs)
    return joint_probs


def generate_data_domains(domain, m, n, r):
    D = np.random.choice(a=[0, 1], size=(m, n))
    if domain == "parity":
        p_flip = 0.1
        D[:, n - 1] = (D[:, n - 1] + np.random.choice(a=[0, 1], size=m, p=[1 - p_flip, p_flip])) % 2
    elif domain == "ParSum":
        fx = sum([(-1) ** (D[:, 2 * i] + D[:, 2 * i + 1]) for i in range(int(n / 2))])
        py_x = 1 / (1 + np.exp(-fx))
        for i in range(m):
            D[i, -1] = np.random.choice(a=[0, 1], size=1, p=[1 - py_x[i], py_x[i]])[0]
    else:
        D = np.random.choice(a=[i for i in range(r)], size=(m, n))

    return D


def random_adj_generator(n_nodes, max_unions=None):
    adj_matrix = np.zeros((n_nodes, n_nodes))
    if not max_unions:
        max_unions = n_nodes - 1
    for r in range(n_nodes):
        n_unions = np.random.randint(1, max_unions)
        children = np.array([np.random.randint(0, n_nodes) for _ in range(n_unions)])
        adj_matrix[r, children] = 1
    return adj_matrix


def binarize_dataset(data, n_bins):
    data_binned = data.copy()
    for v_ix in range(data.shape[1]):
        d = data[:, v_ix]
        data_binned[:, v_ix] = np.digitize(d, np.arange(np.min(d), np.max(d), (np.max(d) - np.min(d)) / n_bins)[1:])
    return data_binned


def generate_topology(topology_type, n_users):
    """
    Generate fully connected topologies between the available ones or random
    """
    adj_matrix = None
    if topology_type == 'star':
        adj_matrix = np.zeros((n_users, n_users))
        adj_matrix[0, 1:] = 1
        adj_matrix[1:, 0] = 1
    elif topology_type == 'line':
        adj_matrix = np.eye(n_users)
        adj_matrix = np.hstack([adj_matrix[:, -1][..., None], adj_matrix[:, :-1]])
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = adj_matrix + adj_matrix.T - np.diag(np.diag(adj_matrix))

    elif topology_type == 'circle':
        adj_matrix = np.eye(n_users)
        adj_matrix = np.hstack([adj_matrix[:, -1][..., None], adj_matrix[:, :-1]])
        adj_matrix = np.triu(adj_matrix)
        adj_matrix = adj_matrix + adj_matrix.T - np.diag(np.diag(adj_matrix))
        adj_matrix[n_users - 1, 0] = 1
        adj_matrix[0, n_users - 1] = 1
    else:
        # random topology
        adj_matrix = random_adj_generator(n_users, n_users - 1)
    return adj_matrix


def generate_exec_sequence(seq_type, size, n_users):
    exec_sequence = np.array([])
    if seq_type == 'uniform':
        l = np.random.choice(range(n_users), size, replace=True)
        exec_sequence = l.tolist()
    elif seq_type == 'perm_comb':
        while len(exec_sequence) < size:
            exec_sequence = np.append(exec_sequence, np.random.permutation(n_users))
        exec_sequence = exec_sequence.astype(int).tolist()
    return exec_sequence
