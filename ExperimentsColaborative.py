import os

import numpy as np
from tqdm import tqdm

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbs
from multiprocessing import Pool

from GenerateData import generate_data_domains, random_adj_generator, binarize_dataset, generate_topology, \
    generate_exec_sequence
from Utils import loadSupervisedData, plot_results
from network.Graph import Graph
from network.Utils import generate_communication_sequence


def main(export_path, adj_matrices, classifiers, datasets, initialize_stats, type_com_queue='ordered',
         show_network=True, base_adj_path='./data/network_definitions/', base_dataset_path='./data/', repetitions=5,
         prob_repetition=[1], num_times=[1]):

    for d in datasets:
        dpath = base_dataset_path + d + '.csv'
        # Read the data
        D, card = loadSupervisedData(dataName=dpath, skipHeader=0, bins=7)
        card_x, card_y = card[:-1], card[-1]
        m = D.shape[0]
        percTrain = 0.7
        for a in adj_matrices:
            adj_path = base_adj_path + a + '.txt'
            adj_matrix = np.loadtxt(adj_path)
            for classifier in classifiers:
                # if a == 'Scenario9.txt':
                #     type_com_queue = 'prob_dist'
                # else:
                #     type_com_queue = 'ordered'
                results_global = pd.DataFrame(None, columns=['seed', 'n_train', 'n_test', 'n_local_train',
                                                             'n_users', 'score_name', 'score', 'time', 'id_user',
                                                             'n_local_iterations', 'policy'])

                exp_path = f"{export_path}/{d}/{a}/{classifier}/"
                check_path_create(exp_path)
                if show_network:
                    from network import Utils
                    Utils.show_graph(adj_matrix, export_path=exp_path + 'Network.pdf')

                for q in prob_repetition:
                    for nt in num_times:
                        for rep in tqdm(range(repetitions), desc='Iteration', ascii=True):
                            # Training test split
                            np.random.seed(rep)
                            perm = np.random.permutation(m)
                            m_train = int(percTrain * m)
                            train_data = D[perm[:m_train], :]
                            test_data = D[perm[m_train:], :]

                            user_rep = 20
                            n_users_exec = len(adj_matrix) * user_rep
                            if type_com_queue == 'ordered':
                                exec_sequence = np.ravel(np.repeat([np.arange(0, len(adj_matrix))], user_rep, axis=0)).tolist()
                            elif type_com_queue == 'perm_comb':
                                ref_permu = np.random.permutation(len(adj_matrix))
                                exec_sequence = np.ravel(np.repeat([ref_permu], user_rep, axis=0)).tolist()
                                # exec_sequence = generate_exec_sequence(seq_type='perm_comb', size=n_users_exec,
                                #                                        n_users=len(adj_matrix))
                            elif type_com_queue == 'prob_dist':
                                exec_sequence = np.random.choice(np.arange(0, 5), p=[0.225, 0.225, 0.1, 0.225, 0.225],
                                                                 size=n_users_exec).tolist()
                                plt.hist(exec_sequence)
                                plt.show()
                                plt.clf()
                            elif type_com_queue == 'intermediate_perm_random':
                                exec_sequence = generate_communication_sequence(size=n_users_exec,
                                                                                num_nodes=len(adj_matrix),
                                                                                num_times=nt,
                                                                                prob_stability_q=q)

                            else:  # Random
                                exec_sequence = np.random.randint(0, len(adj_matrix), n_users_exec)
                                # plt.hist(exec_sequence)
                                # plt.show()
                                # plt.clf()

                            exec_sequence_str = ','.join(str(x) for x in exec_sequence)
                            g = Graph(adj_matrix, 'info', train_data, classifier, exec_sequence, card_x=card_x, card_y=card_y,
                                      seed=rep)
                            results = g.start(test_data, initialize_stats=initialize_stats)
                            results = pd.DataFrame(results, columns=['seed', 'n_train', 'n_test', 'n_local_train', 'n_users',
                                                                     'score_name', 'score', 'time', 'id_user',
                                                                     'n_local_iterations',
                                                                     'policy'])
                            results['data_name'] = d
                            results['network_topology'] = a
                            results['type_sequence'] = type_com_queue
                            results['BN_Structure'] = classifier
                            results['num_times'] = nt
                            results['p_repetition'] = q
                            results['exec_sequence'] = exec_sequence_str
                            results_global = pd.concat([results_global, results])

                # Check if the path is created
                results_global.to_csv(exp_path + 'resultsUniform.csv', sep=',', decimal='.', index=False)
                print(f"The results have been exported to: {exp_path}\n")


def check_path_create(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


def to_latex(initialize_stats, datasets, classifiers, adj_matrices, grid=False, grid_name=None, permu=None):
    for d in datasets:
        for classifier in classifiers:
            out_string = '''\\documentclass[]{{article}}  
            \\usepackage{{babel}}  
            \\usepackage[utf8]{{inputenc}}
            \\usepackage{{graphicx}}
            \\usepackage[caption=false]{{subfig}}
            \\usepackage{{float}}
            \\usepackage[margin=2cm]{{geometry}}

            % Title Page
            '''.format(length='multi-line', ordinal='second')
            out_string += f'\\title{{Experiments {d} - {initialize_stats}}} '
            out_string += '''
            \\author{{Ander Carreño}}
            \\date{{\\today}}

            \\begin{{document}}\n
            \\maketitle\n'''.format(length='multi-line', ordinal='second')
            for a in adj_matrices:
                exp_path = f"{export_path}{d}/{a}/{classifier}/"
                # results_global = pd.read_csv(exp_path + 'resultsUniform.csv', sep=',')

                out_string += f'\\section{{{a}}}'
                out_string += '''
\\begin{{figure}}[H] 
    \\centering\n'''.format(length='multi-line', ordinal='second')
                if not grid:
                    # To generate the individual results
                    out_string += f'\t\\subfloat[Network]{{\\includegraphics[width=.50\\textwidth]{{{exp_path + "Network.pdf"}}}}}\\hfill\n'
                    out_string += f'\t\\subfloat[Uniform]{{\\includegraphics[width=.50\\textwidth]{{{exp_path + "Uniform.pdf"}}}}}\n'
                    out_string += f'\\caption{{{permu}}}'
                else:
                    # To generate the GRID SEARCH results
                    out_string += f'\t\\subfloat[Network]{{\\includegraphics[width=.50\\textwidth]{{{exp_path + "Network.pdf"}}}}}\\hfill\n'
                    out_string += f'\t\\subfloat[Grid Search]{{\\includegraphics[width=\\textwidth]{{{exp_path + grid_name}}}}}'
                    out_string += f'\\caption{{Grid search}}'


                out_string += ''' 
\\end{{figure}}\n\n'''.format(length='multi-line', ordinal='second')

            out_string += '\end{document}'

            out_path = f"{d}_{grid_name}.tex"
            text_file = open(out_path, "w")
            text_file.write(out_string)
            text_file.close()

            os.system(f"pdflatex {out_path}")
            print("The PDF has been generated in the root directory of the project.")


if __name__ == '__main__':
    # Parameters
    export_path = './Results/ExperimentsColaborative/'
    adj_matrices = ['Scenario1', 'Scenario2', 'Scenario3', 'Scenario4', 'Scenario5',
                    'Scenario6', 'Scenario7', 'Scenario8', 'Scenario9']
    # adj_matrices = ['Scenario1', 'Scenario2']
    type_com_queue = 'intermediate_perm_random'
    classifiers = ['NB']
    datasets = ['glass']
    prob_repetition = [0, 0.5, 1]
    num_times = [1, 2, 4]
    initialize_stats = 'uniform'
    # initialize_stats = 'ML_local'

    # Run the experiment
    main(export_path=export_path,
         adj_matrices=adj_matrices,
         classifiers=classifiers,
         datasets=datasets,
         type_com_queue=type_com_queue,
         prob_repetition=prob_repetition,
         num_times=num_times,
         initialize_stats=initialize_stats)


    logloss = [True, False]
    grid = True
    # Plot the results
    for d in datasets:
        for a in tqdm(adj_matrices, desc='Scenario', ascii=True):
            for classifier in classifiers:
                for l in logloss:
                    exp_path = f"{export_path}{d}/{a}/{classifier}/"
                    results_global = pd.read_csv(exp_path + 'resultsUniform.csv', sep=',')
                    if l:
                        ixs = [_ in ['logloss_ML_TEST', 'logloss_ML_TRAIN', 'logloss_local', 'logloss_global', 'logloss_test'] for _ in results_global['score_name'].values]
                        grid_name = 'grid_logloss'
                    else:
                        ixs = [_ in ['CLL_local', 'CLL_global', 'CLL_test', 'CLL_ML_TEST', 'CLL_ML_TRAIN'] for _ in results_global['score_name']]
                        grid_name = 'grid_CLL'
                    if grid:
                        g = sbs.FacetGrid(results_global.loc[ixs, :], col="num_times", row="p_repetition", hue="score_name")
                        g.map(sbs.lineplot, "time", "score")
                        g.add_legend()
                        g.savefig(exp_path + grid_name + '.pdf', format='pdf')
                        plt.clf()
                    else:
                        plot_results(results_global, export_path=exp_path + 'Uniform.pdf',
                                     title=initialize_stats + ' ' + a + ' ' + classifier)
                        # plot_results(results_global, title=initialize_stats + ' ' + a + ' ' + classifier)

    to_latex(initialize_stats, datasets, classifiers, adj_matrices, grid, 'grid_CLL')
    to_latex(initialize_stats, datasets, classifiers, adj_matrices, grid, 'grid_logloss')

