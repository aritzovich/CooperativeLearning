import os

import numpy as np
from tqdm import tqdm

import pandas as pd
from matplotlib import pyplot as plt

from GenerateData import generate_data_domains, random_adj_generator, binarize_dataset, generate_topology, \
    generate_exec_sequence
from Utils import loadSupervisedData, plot_results
from network.Graph import Graph


def main(export_path, adj_matrices, classifiers, datasets, type_com_queue='ordered', show_network=True,
         base_adj_path='./data/network_definitions/', base_dataset_path='./data/', repetitions=10):
    for d in datasets:
        dpath = base_dataset_path + d
        data, card = loadSupervisedData(dpath, sep=',', skipHeader=0, bins=3)
        card_x, card_y = card[:-1], card[-1]
        perm = np.random.permutation(len(data))
        ix = np.ceil(len(data) * 0.3).astype(int)
        test_data = data[perm[:ix], :]
        train_data = data[perm[ix:], :]
        for a in adj_matrices:
            adj_path = base_adj_path + a
            adj_matrix = np.loadtxt(adj_path)
            for classifier in classifiers:
                if a == 'Scenario9.txt':
                    type_com_queue = 'prob_dist'
                else:
                    type_com_queue = 'ordered'

                results_global = pd.DataFrame(None, columns=['seed', 'n_train', 'n_test', 'n_local_train',
                                                             'n_users', 'score_name', 'score', 'time', 'id_user',
                                                             'n_local_iterations', 'policy'])

                exp_path = export_path + a + '/' + classifier + '/'
                check_path_create(exp_path)
                if show_network:
                    from network import Utils
                    Utils.show_graph(adj_matrix, export_path=exp_path + 'Network.pdf')

                for rep in tqdm(range(repetitions), desc='Iteration'):
                    np.random.seed(rep)

                    user_rep = 20
                    n_users_exec = len(adj_matrix) * user_rep
                    if type_com_queue == 'ordered':
                        exec_sequence = np.ravel(np.repeat([np.arange(0, len(adj_matrix))], user_rep, axis=0)).tolist()
                    elif type_com_queue == 'perm_comb':
                        exec_sequence = generate_exec_sequence(seq_type='perm_comb', size=n_users_exec,
                                                               n_users=len(adj_matrix))
                    elif type_com_queue == 'prob_dist':
                        exec_sequence = np.random.choice(np.arange(0, 5), p=[0.225, 0.225, 0.1, 0.225, 0.225], size=n_users_exec).tolist()
                        plt.hist(exec_sequence)
                        plt.show()
                        plt.clf()
                    else:  # Random
                        exec_sequence = np.random.randint(0, len(adj_matrix), n_users_exec)
                        plt.hist(exec_sequence)
                        plt.show()
                        plt.clf()

                    g = Graph(adj_matrix, 'info', train_data, classifier, exec_sequence, card_x=card_x, card_y=card_y,
                              seed=rep)
                    results = g.start(test_data)
                    results = pd.DataFrame(results, columns=['seed', 'n_train', 'n_test', 'n_local_train', 'n_users',
                                                             'score_name', 'score', 'time', 'id_user',
                                                             'n_local_iterations',
                                                             'policy'])
                    results['data_name'] = d
                    results['network_topology'] = a
                    results['type_sequence'] = type_com_queue
                    results['BN_Structure'] = classifier
                    results_global = pd.concat([results_global, results])

                # Check if the path is created
                results_global.to_csv(exp_path + 'resultsUniform.csv', sep=',', decimal='.', index=False)
                print(f"The results have been exported to: {exp_path}\n")


def check_path_create(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


def to_latex():
    out_string = '''\\documentclass[]{{article}}  
\\usepackage{{babel}}  
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage[caption=false]{{subfig}}
\\usepackage{{float}}
\\usepackage[margin=2cm]{{geometry}}

% Title Page
\\title{{Experiments Collaborative Learning}}
\\author{{Ander Carreño}}
\\date{{\\today}}

\\begin{{document}}\n
\\maketitle\n'''.format(length='multi-line', ordinal='second')

    for a in adj_matrices:
        for d in datasets:
            for classifier in classifiers:
                exp_path = export_path + a + '/' + classifier + '/'
                out_string += f'\\section{{{a}}}'
                out_string += '''
\\begin{{figure}}[H] 
    \\centering\n'''.format(length='multi-line', ordinal='second')
                out_string += f'\t\\subfloat[Network]{{\\includegraphics[width=.50\\textwidth]{{{exp_path + "Network.pdf"}}}}}\\hfill\n'
                out_string += f'\t\\subfloat[Uniform]{{\\includegraphics[width=.50\\textwidth]{{{exp_path + "Uniform.pdf"}}}}}'
                out_string += ''' 
    \\caption{{Results of the experiments.}}
\\end{{figure}}\n\n'''.format(length='multi-line', ordinal='second')

    out_string += '\end{document}'

    text_file = open("report.tex", "w")
    text_file.write(out_string)
    text_file.close()

    os.system("pdflatex report.tex")
    print("The PDF has been generated in the root directory of the project.")


if __name__ == '__main__':
    # Parameters
    export_path = './Results/ExperimentsColaborative/'
    adj_matrices = ['Scenario1.txt', 'Scenario2.txt', 'Scenario3.txt', 'Scenario4.txt', 'Scenario5.txt',
                    'Scenario6.txt', 'Scenario7.txt', 'Scenario8.txt', 'Scenario9.txt', 'Scenario10.txt']
    type_com_queue = 'ordered'
    classifiers = ['NB']
    datasets = ['iris.csv']

    # Run the experiment
    main(export_path, adj_matrices, classifiers, datasets, type_com_queue=type_com_queue)

    # Plot the results
    for a in adj_matrices:
        for d in datasets:
            for classifier in classifiers:
                exp_path = export_path + a + '/' + classifier + '/'
                results_global = pd.read_csv(exp_path + 'resultsUniform.csv', sep=',')
                plot_results(results_global, export_path=exp_path + 'Uniform.pdf',
                             title='Uniform ' + a + ' ' + classifier)
                # plot_results(results_global, title='Uniform ' + a + ' ' + classifier)
    to_latex()
