import sys
import os
import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import random

def mhc_type_comp_logos():
    # two sample logos - MHCI vs MHCII peptides
    datafile = '/home/dsi/speingi/PycharmProjects/ERGO-II_dev/data/VDJDB_complete.tsv'
    data = pd.read_csv(datafile, engine='python', sep='\t')
    print(len(data))
    mhci = data[data['MHC class'] == 'MHCI']
    mhcii = data[data['MHC class'] == 'MHCII']
    print(len(mhci))
    print(len(mhcii))
    # MEGA problem - Two Sample Logos askes for pos/neg sequences of the same length
    # but mhcii peptides are always longer than 9-10 mers mhci peptides.
    pass


def linear_regression_coefficients(test_type):
    if test_type == 'tpp':
        # matrix with flags for what features were used
        # tpp type, alpha, vj, mhc, t-type
        a = np.zeros((3, 5))
        # tpp
        a[:, 0] = [1, 2, 3]
        # beta
        a0 = a.copy()
        # beta, vj
        a1 = a.copy()
        a1[:, 2] = 1
        # beta, mhc
        a2 = a.copy()
        a2[:, 3] = 1
        # beta, t-type
        a3 = a.copy()
        a3[:, 4] = 1
        # beta, alpha
        a4 = a.copy()
        a4[:, 1] = 1
        # beta, alpha, vj
        a5 = a.copy()
        a5[:, 1] = 1
        a5[:, 2] = 1
        # beta, alpha, vj, mhc
        a6 = a.copy()
        a6[:, 1] = 1
        a6[:, 2] = 1
        a6[:, 3] = 1
        # beta, alpha, vj, mhc, t-type
        a7 = a.copy()
        a7[:, 1] = 1
        a7[:, 2] = 1
        a7[:, 3] = 1
        a7[:, 4] = 1
        X = np.concatenate((a0, a1, a2, a3, a4, a5, a6, a7), axis=0)
        # print(X)
        with open('mcpas_tpp_results.csv', 'r') as file:
            y = []
            y2 = []
            file.readline()
            for line in file:
                line = line.strip().split('\t')
                y.append(eval(line[0]))
                y2.append(eval(line[1]))
        assert X.shape[0] == len(y)
        reg = LinearRegression().fit(X, y)
        reg2 = LinearRegression().fit(X, y2)
        labels = ['tpp test', 'alpha', 'vj', 'mhc', 't-type']
        assert len(labels) == len(reg.coef_.reshape(-1, 1))
        plot_data = np.concatenate((reg.coef_.reshape(-1, 1),
                                    reg2.coef_.reshape(-1, 1)), axis=1)
        print(plot_data)
        # exit()
        bplot = plt.boxplot(plot_data.T,
                             vert=True,  # vertical box alignment
                             patch_artist=True, # fill with color
                             labels=labels) # will be used to label x-ticks
        plt.title('TPP Linear Regression Coefficients')
        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen']
        bplot['boxes'][0].set_facecolor('lightblue')
        plt.show()
    if test_type == 'spb':
        peptides = ['LPRRSGAAGA', 'GILGFVFTL', 'NLVPMVATV', 'GLCTLVAML', 'SSYRRPVGI']
        # matrix with flags for what features were used
        # alpha, vj, mhc, t-type
        a = np.zeros((5, 4))
        for i in range(1, 5):
            a[i, 0:i] = 1
        X = a
        with open('mcpas_spb_results.csv', 'r') as file:
            y = []
            y2 = []
            for i, line in enumerate(file):
                line = line.strip().split('\t')
                if i % 2:
                    y.append([eval(t) for t in line])
                else:
                    y2.append([eval(t) for t in line])
        for i, pep in enumerate(peptides):
            reg = LinearRegression().fit(X, y[i])
            reg2 = LinearRegression().fit(X, y2[i])
            labels = ['alpha', 'vj', 'mhc', 't-type']
            assert len(labels) == len(reg.coef_.reshape(-1, 1))
            plot_data = np.concatenate((reg.coef_.reshape(-1, 1),
                                        reg2.coef_.reshape(-1, 1)), axis=1)
            print(plot_data)
            # exit()
            bplot = plt.boxplot(plot_data.T,
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
                                labels=labels)  # will be used to label x-ticks
            plt.title('SPB (' + pep + ') Linear Regression Coefficients')
            # fill with colors
            plt.show()
    pass


if __name__ == '__main__':
    # mhc_type_comp_logos()
    # linear_regression_coefficients('tpp')
    linear_regression_coefficients('spb')
