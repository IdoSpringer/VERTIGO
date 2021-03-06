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


def spb_linear_regression_coefficients(results_file):
    results = pd.read_csv(results_file)
    peptides = results['peptide']
    avg = []
    for j, peptide in enumerate(peptides[:-1]):
        # alpha, vj, mhc, t-type
        a = np.zeros((5, 4))
        for i in range(1, 5):
            a[i, 0:i] = 1
        X = a
        pep_results = results.iloc[j][1:]
        ae_results = [pep_results[i] for i in range(len(pep_results)) if i % 2]
        lstm_results = [pep_results[i] for i in range(len(pep_results)) if not i % 2]
        reg_ae = LinearRegression().fit(X, ae_results)
        reg_lstm = LinearRegression().fit(X, lstm_results)
        labels = ['TCR' + r'$\beta$', 'TCR' + r'$\alpha$', 'V, J', 'MHC', 'T-Type']
        ae_plot = np.concatenate([np.array([reg_ae.intercept_ - 0.5]), reg_ae.coef_])
        lstm_plot = np.concatenate([np.array([reg_lstm.intercept_ - 0.5]), reg_lstm.coef_])
        assert len(labels) == len(ae_plot.reshape(-1, 1))
        # plot_data = np.concatenate((reg_ae.coef_.reshape(-1, 1),
        #                             reg_lstm.coef_.reshape(-1, 1)), axis=1)
        plot_data = np.concatenate((ae_plot.reshape(-1, 1),
                                    lstm_plot.reshape(-1, 1)), axis=1)
        avg.append(plot_data)
        plt.subplot(5, 4, j + 1)
        bplot = plt.boxplot(plot_data.T,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=labels),
        colors = ['dodgerblue', 'silver', 'seagreen', 'goldenrod', 'tomato']
        for patch, color in zip(bplot[0]['boxes'], colors):
            patch.set_facecolor(color)
        # axes = plt.gca()
        # axes.set_ylim([-0.05, 0.3])
        plt.title(str(peptide), fontdict={'fontsize': 8})
        # plt.xlabel(str(peptide))
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        if j == 8:
            plt.ylabel('AUC Contribution', fontdict={'fontsize': 14})
    avg_plot_data = np.average(avg, axis=0)
    plt.subplot(5, 4, 20)
    bplot = plt.boxplot(avg_plot_data.T,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=labels),
    colors = ['silver', 'orchid', 'goldenrod', 'deepskyblue']
    for patch, color in zip(bplot[0]['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Average', fontdict={'fontsize': 8})
    # plt.xlabel(str(peptide))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # plt.tight_layout(pad=0, rect=(0,0,1,1))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.suptitle('McPAS SPB Linear Regression Coefficients', fontsize=16)
    plt.show()


def tpp_linear_regression_coefficients(results_file):
    results = pd.read_csv(results_file)
    # tpp-i, tpp-ii, tpp-iii, alpha, vj, mhc, t-type
    a = np.zeros((3*8, 7))
    a[:, 0] = [1, 0, 0] * 8
    a[:, 1] = [0, 1, 0] * 8
    a[:, 2] = [0, 0, 1] * 8
    a[3:6, 4] = 1
    a[6:9, 5] = 1
    a[9:12, 6] = 1
    a[12:15, 3] = 1
    a[15:18, 3:5] = 1
    a[18:21, 3:6] = 1
    a[21:24, 3:7] = 1
    X = a
    results = pd.read_csv(results_file, sep='\t')
    m_ae_results = results['mcpas_ae']
    m_lstm_results = results['mcpas_lstm']
    reg_m_ae = LinearRegression().fit(X, m_ae_results)
    reg_m_lstm = LinearRegression().fit(X, m_lstm_results)

    ae_plot = np.concatenate([reg_m_ae.coef_[:3], np.array([reg_m_ae.intercept_ - 0.5]), reg_m_ae.coef_[3:]])
    print(ae_plot)
    lstm_plot = np.concatenate([reg_m_lstm.coef_[:3], np.array([reg_m_lstm.intercept_ - 0.5]), reg_m_lstm.coef_[3:]])
    print(lstm_plot)
    labels = ['TPP-I', 'TPP-II', 'TPP-III', 'TCR' + r'$\beta$', 'TCR' + r'$\alpha$', 'V, J', 'MHC', 'T-Cell Type']
    assert len(labels) == len(ae_plot.reshape(-1, 1))
    bars1 = plt.bar(range(3), ae_plot[:3], width=0.2,
            label='AE', color='indigo')
    bars2 = plt.bar(range(3, 8), ae_plot[3:], width=0.2,
                   label='AE', color='dodgerblue')
    bars3 = plt.bar([x + 0.2 for x in range(3)], lstm_plot[:3],
            width=0.2, label='LSTM', color='indianred')
    bars4 = plt.bar([x + 0.2 for x in range(3, 8)], lstm_plot[3:],
                   width=0.2, label='LSTM', color='salmon')
    plt.legend()
    first_legend = plt.legend(handles=[bars1, bars3], loc='upper left')
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[bars2, bars4], loc='upper right')
    plt.xticks([x+0.2 for x in range(len(labels))], labels)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(2.6, color='black', lw=0.5, dashes=(10, 6))
    plt.title('McPAS TPP Linear Regression Coefficients', fontdict={'fontsize': 14})
    plt.ylabel('AUC Contribution', fontdict={'fontsize': 14})
    colors = ['lightblue'] * 3 + ['lightgreen'] * 4
    plt.tight_layout()
    print(reg_m_ae.coef_)
    print(reg_m_lstm.coef_)
    print('The standart error between the models is bounded by %.3f' %
          max(np.std([ae_plot, lstm_plot], axis=0)))
    plt.show()


if __name__ == '__main__':
    # mhc_type_comp_logos()
    # linear_regression_coefficients('tpp')
    # linear_regression_coefficients('spb')
    # spb_linear_regression_coefficients('mcpas_spb_results.csv')
    tpp_linear_regression_coefficients('tpp_results.csv')
