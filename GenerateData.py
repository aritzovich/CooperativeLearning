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
        D[:, n-1] = (D[:, n-1]+np.random.choice(a=[0, 1], size=m, p=[1-p_flip, p_flip])) % 2
    elif domain == "ParSum":
        fx = sum([(-1)**(D[:, 2*i]+D[:, 2*i+1]) for i in range(int(n/2))])
        py_x = 1/(1 + np.exp(-fx))
        for i in range(m):
            D[i, -1] = np.random.choice(a=[0,1], size= 1, p=[1-py_x[i], py_x[i]])[0]
    else:
        D = np.random.choice(a=[i for i in range(r)], size=(m, n))

    return D

