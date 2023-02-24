import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def loadSupervisedData(dataName, sep=',', skipHeader= 0,  classInd=None, maxDiscVals=5, bins=None):
    '''
    Load a supervised classification data in csv format: each row contains an instance (case), each column corresponds
    to the values of random variable. The values included in each row are separated by a string given by the parameter
    sep, e.g., ",". The column of the class variable is given by classInd, and in case of classInd==None the class
    variable corresponds to the last column.

    A variable is considered discrete when i) the values are strings, or 2) the number of values it takes is smaller or
    equal to maxDiscVals.

    :param dataName: name of the file containing the data set
    :param sep: the separator of the CSV. By default ','
    :param skipHeader: the header lines that are going to be skyped. By default 0
    :param classInd: index of the class variable. By default (=None) the last variable
    :param maxDiscVals: a continuous variable is condidered discrete when the number of different values is lower or
    equal to maxDiscVals
    :param bins: Decides whether or not to discretize using equal frequency the continuous variables and the number of
    intervals.  If it is a positive integer represents the number of bins. By default variables are not discretized

    :return: the data set np.array(numCases x numVars), and the cardinality of the variables where continuous have
    np.inf cardinality
    '''

    text = np.genfromtxt(dataName, dtype= str, delimiter=sep, skip_header= skipHeader)
    (m, n) = text.shape
    if classInd is None:
        classInd= n-1

    # Determine the nature of the variables (categorical or continuous)
    card = np.zeros(n,dtype=int)
    for i in range(n):
        if i != classInd:
            vals= np.unique(text[:, i])
            if str.isalpha(text[0, i]):
                card[i]= len(vals)

            elif str.isdigit(text[0, i]):  # if all characters in the string are alphabetic or there is at least one character
                if len(vals)<= maxDiscVals:
                    card[i] = len(vals)
                else:
                    card[i] = np.iinfo(np.int32).max
            else:
                if len(vals) <= maxDiscVals:
                    card[i] = len(vals)
                else:
                    card[i] = np.iinfo(np.int32).max
        else:
            card[i]= len(np.unique(text[:,i]))

    if bins is not None:
        data = np.zeros((m,n),dtype=int)
    else:
        data = np.zeros((m,n),dtype=np.float)

    for i in range(n):
        if card[i]== np.iinfo(int).max:
            data[:,i] = np.array([np.float(x) for x in text[:, i]])
        else:
            data[:,i] = np.unique(text[:, i], return_inverse=True)[1]


    if bins is not None:
        #Discretize continuous data using equal frequency
        for i in range(n):
            if card[i]== np.iinfo(np.int32).max:
                ordered = np.sort(data[:,i])
                cut = [ordered[int((j + 1) * m / bins) - 1] for j in range(bins)]
                cut[bins - 1] = ordered[m - 1]
                for j in range(m):
                    for k in range(bins):
                        if data[j,i] <= cut[k]:
                            break
                    data[j,i] = k

                card[i]= bins

    return data,card


def missing_value_imputation(df, imputation_type="mode", max_distinct_values=5):
    """
    Imputes the missing values of a dataset with the specified imputation method
    :param df: The dataframe that is going to be filled
    :param imputation_type: The imputation method. Currently only 'mode'
    :param max_distinct_values: Max distinct values that a feature can take so that it can be considered as categorical
    :return: The filled dataframe
    """
    m, n = df.shape
    for c in range(n):
        unique_vals = np.unique(df[:, c])
        n_distinct_values = len(unique_vals)
        ix = df[:, c] == df[:, c]
        if np.sum(ix) > 0:  # If there are missing values in this column
            if n_distinct_values > max_distinct_values:
                # It is considered continuous
                mean = np.nanmean(np.df[:, c])
                df[ix, c] = mean
            else:
                # It is considered categorical
                counts = [np.nansum(df[:, c] == val) for val in unique_vals]
                mode = unique_vals[np.argmax(counts)]
                df[ix, c] = mode
    return df


def plot_results(df, theme='darkgrid', export_path=None):
    sns.set_theme(style=theme)
    sns.lineplot(x="time", y="score",
                 hue="score_name", style="BN_Structure",
                 data=df)
    if export_path:
        plt.savefig(export_path, format='pdf')
    else:
        plt.show()
