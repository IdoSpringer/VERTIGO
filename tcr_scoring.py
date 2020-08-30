import pandas as pd
from Loader import SinglePeptideDataset, get_index_dicts
from Trainer import ERGOLightning
import pickle
import torch
from torch.utils.data import DataLoader
from argparse import Namespace
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join


def read_repertoire(file):
    print('Reading %s ...' % file)
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    def fixv(v):
        if '-' in v:
            return v.split('-')[0] + '-0' + v.split('-')[-1]
        else:
            return v
    data = pd.read_csv(file, engine='python', sep='\t')
    samples = []
    tcrs = data['AA. Seq. CDR3'].tolist()
    vs = data['All V hits'].tolist()
    js = data['All J hits'].tolist()
    freq = data['Clone fraction'].tolist()
    for tcr, v, j, f in zip(tcrs, vs, js, freq):
        if invalid(tcr):
            continue
        if len(tcr) < 8:
            continue
        v = v.split(',')[0]
        v = v.split('*')[0]
        v = fixv(v)
        j = j.split(',')[0]
        j = j.split('*')[0]
        samples.append({'tcrb': tcr,
                        'tcra': 'UNK',
                        'va': 'UNK',
                        'ja': 'UNK',
                        'vb': v,
                        'jb': j,
                        'mhc': 'UNK',
                        't_cell_type': 'UNK',
                        'freq': f,
                        'sign': 0})
    print('Done reading file')
    return samples


def load_model(hparams, checkpoint_path):
    model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def score(samples, peptide, model, hparams, threshold=0.9, detect_tcrs=False):
    train_file = 'Samples/' + model.dataset + '_train_samples.pickle'
    with open(train_file, 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    testset = SinglePeptideDataset(samples, train_dicts, peptide,
                                   force_peptide=True, spb_force=False)
    loader = DataLoader(testset, batch_size=2048, shuffle=False, num_workers=10,
                        collate_fn=lambda b: testset.collate(b, tcr_encoding=hparams.tcr_encoding_model,
                                                             cat_encoding=hparams.cat_encoding))
    yf_tcrs = []
    model.eval()
    scores = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(batch_idx)
            outputs = model.validation_step(batch, batch_idx)['y_hat']
            scores.extend(outputs.tolist())
            if detect_tcrs:
                indicies = outputs > threshold
                pos_tcrb = batch[1][indicies]
                for tcr_tensor in pos_tcrb:
                    tcr = decode_tcr(tcr_tensor)
                    print(tcr)
                    yf_tcrs.append(tcr)
            if batch_idx >= 10:
                break
    return scores, yf_tcrs


def frequency_score_scatter(samples, peptide, model, hparams):
    train_file = 'Samples/' + model.dataset + '_train_samples.pickle'
    with open(train_file, 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    batch_size = 2048
    testset = SinglePeptideDataset(samples, train_dicts, peptide,
                                   force_peptide=True, spb_force=False)
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10,
                        collate_fn=lambda b: testset.collate(b, tcr_encoding=hparams.tcr_encoding_model,
                                                             cat_encoding=hparams.cat_encoding))
    model.eval()
    scores = []
    freqs = []
    i = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(batch_idx)
            outputs = model.validation_step(batch, batch_idx)['y_hat']
            scores.extend(outputs.tolist())
            freqs += [np.log(s['freq']) for s in samples[i:i+batch_size]]
            i += batch_size
            assert len(scores) == len(freqs)
            if batch_idx >= 5:
                break
    return scores, freqs


def decode_tcr(tcr_tensor):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    xtoa = {index: amino for index, amino in enumerate(['PAD'] + amino_acids + ['X'])}
    tcr = ''
    for l in tcr_tensor:
        letter = xtoa[l.tolist().index(1) + 1]
        tcr += letter
        if letter == 'X':
            break
    tcr = tcr[:-1]
    return tcr


def main(rep, func):
    # get model file from version
    # model_dir = 'yellow_fever/YF_models'
    # checkpoint_path = os.path.join(model_dir, 'version_' + version, 'checkpoints')
    # files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
    # checkpoint_path = os.path.join(checkpoint_path, files[0])
    checkpoint_path = 'paper_models/version_1mej/checkpoints/_ckpt_epoch_9.ckpt'
    # get args from version
    # args_path = os.path.join(model_dir, version, 'meta_tags.csv')
    args_path = 'ERGO-II_paper_logs/paper_models/version_1mej/meta_tags.csv'
    with open(args_path, 'r') as file:
        lines = file.readlines()
        args = {}
        for line in lines[1:]:
            key, value = line.strip().split(',')
            if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
                args[key] = value
            else:
                args[key] = eval(value)
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    yf_peptide = 'LLWNGPMAV'
    if func == 'hist':
        colors = ['r', 'g', 'b', 'y', 'c']
        timepoints = ['-1', '0', '7', '15', '45']
        for i, time in enumerate(timepoints):
            file = '_'.join([rep, time, 'F1']) + '.txt'
            samples = read_repertoire(file)
            scores, yf_tcrs = score(samples, yf_peptide, model, hparams)
            plt.hist(scores, 100, facecolor=colors[i], alpha=0.2, label=timepoints[i], histtype='bar')
        plt.title('ERGO Score Histogram for Yellow Fever Peptide')
        plt.xlabel('ERGO Score')
        plt.ylabel('Number of TCRs')
        plt.legend()
        plt.show()
    if func == 'scatter':
        # file = '_'.join([rep, '15', 'F1']) + '.txt'
        # samples = read_repertoire(file)
        # scores, freqs = frequency_score_scatter(samples, yf_peptide, model, hparams)
        # plt.scatter(scores, freqs)
        # plt.show()
        colors = ['r', 'g', 'b', 'y', 'c']
        timepoints = ['-1', '0', '7', '15', '45']
        for i, time in enumerate(timepoints):
            file = '_'.join([rep, time, 'F1']) + '.txt'
            samples = read_repertoire(file)
            scores, freqs = frequency_score_scatter(samples, yf_peptide, model, hparams)
            plt.scatter(scores, freqs, color=colors[i], alpha=0.3, label=timepoints[i])
        plt.title('Frequency and ERGO Yellow Fever Score Scatter')
        plt.xlabel('ERGO Score')
        plt.ylabel('Log Frequency')
        plt.legend()
        plt.show()
    return


if __name__ == '__main__':
    # we use mcpas model with v and j
    # we do not train again with extra weight on yf
    rep = 'yellow_fever/Yellow_fever/' + sys.argv[1]
    main(rep, func='scatter')
    pass

# python tcr_scoring.py P1_0_F1.txt
