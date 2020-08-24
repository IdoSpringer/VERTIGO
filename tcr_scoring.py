import pandas as pd
from Loader import SinglePeptideDataset, get_index_dicts
from Trainer import ERGOYellowFever
import pickle
import torch
from torch.utils.data import DataLoader
from argparse import Namespace
import sys
import os
from os import listdir
from os.path import isfile, join


def read_repertoire(file):
    print('Reading %s ...' % file)
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    data = pd.read_csv(file, engine='python', sep='\t')
    tcrs = data['AA. Seq. CDR3'].tolist()
    tcrs = list(filter(lambda x: len(x) > 8, tcrs))
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    tcrs = list(filter(lambda x: not invalid(x), tcrs))
    print('Done reading file')
    return tcrs


def load_model(hparams, checkpoint_path):
    model = ERGOYellowFever(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def score(tcrs, peptide, model, hparams, threshold=0.9):
    # actually not relevant in this case (without V gene)
    train_file = 'Samples/' + model.dataset + '_train_samples.pickle'
    with open(train_file, 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    samples = []
    for tcr in tcrs:
        samples.append({'tcrb': tcr,
                        'tcra': 'UNK',
                        'va': 'UNK',
                        'ja': 'UNK',
                        'vb': 'UNK',
                        'jb': 'UNK',
                        'mhc': 'UNK',
                        't_cell_type': 'UNK',
                        'sign': 0})
    testset = SinglePeptideDataset(samples, train_dicts, peptide,
                                   force_peptide=True, spb_force=False)
    loader = DataLoader(testset, batch_size=2048, shuffle=False, num_workers=10,
                        collate_fn=lambda b: testset.collate(b, tcr_encoding=hparams.tcr_encoding_model,
                                                             cat_encoding=hparams.cat_encoding))
    yf_tcrs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(batch_idx)
            model.eval()
            outputs = model.validation_step(batch, batch_idx)['y_hat']
            # print(outputs)
            indicies = outputs > threshold
            pos_tcrb = batch[1][indicies]
            for tcr_tensor in pos_tcrb:
                tcr = decode_tcr(tcr_tensor)
                print(tcr)
                yf_tcrs.append(tcr)
    return yf_tcrs


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


def main(version, rep_file):
    # get model file from version
    model_dir = 'yellow_fever/YF_models'
    checkpoint_path = os.path.join(model_dir, 'version_' + version, 'checkpoints')
    files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
    checkpoint_path = os.path.join(checkpoint_path, files[0])
    # get args from version
    args_path = os.path.join(model_dir, version, 'meta_tags.csv')
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
    tcrs = read_repertoire(rep_file)
    yf_tcrs = score(tcrs, yf_peptide, model, hparams)
    return


if __name__ == '__main__':
    version = 'yf1n10xe' + sys.argv[1]
    file = 'yellow_fever/Yellow_fever/' + sys.argv[2]
    main(version, file)
    pass

# python tcr_scoring.py 5 P1_0_F1.txt
