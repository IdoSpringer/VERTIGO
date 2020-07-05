# use ERGO-II (with alpha, V, J) to predoct a part of VDJdb for benchmark.
# we need to check if vdjdb_slim is a test or it is a part of the vdjdb training set.
# if so, lets train on all - vdjdb slim and predict slim
# files in vdjdb folder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Loader import SignedPairsDataset, DiabetesDataset, get_index_dicts
from Models import PaddingAutoencoder, AE_Encoder, LSTM_Encoder, ERGO
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from argparse import ArgumentParser


import pandas as pd
vdjdb_dir = 'VDJdb/'


def check_vdjdb_slim():
    slim_data = pd.read_csv(vdjdb_dir + 'vdjdb.slim.txt', engine='python', sep='\t')
    slim_tcrs = slim_data.cdr3.to_list()
    vdjdb_data = pd.read_csv(vdjdb_dir + 'vdjdb.txt', engine='python', sep='\t')
    vdjdb_tcrs = vdjdb_data.cdr3.to_list()
    print(len(set(vdjdb_tcrs) - set(slim_tcrs))) # = 0
    print(len(set(slim_tcrs) - set(vdjdb_tcrs))) # = 0
    # vdjdb.slim is the same as regular vdjdb... not a test set
    pass


def cross_validation():
    # 5-fold cross validation
    # split vdjdb to five 80% train and 20% test sets
    # we need to know at prediction time what fold to predict for each sample...
    # we need to have access to full database when sampling negative examples
    # tcra and tcrb must be in the same fold
    slim_data = pd.read_csv(vdjdb_dir + 'vdjdb.slim.txt', engine='python', sep='\t')
    slim_data = slim_data.sample(frac=1).reset_index(drop=True)
    # print(slim_data['complex.id'])
    # maybe we cross validate on complex.id (every id=0 is beta only)
    complex = slim_data['complex.id'].to_list()
    # print(complex)
    # count complexes in data
    single_complexes = [g.split(',') for g in complex]
    flat_list = [item for sublist in single_complexes for item in sublist]
    # print(flat_list)
    ab_complexes = [c for c in flat_list if c != '0']
    print('Number of complexes:', len(ab_complexes) // 2 + flat_list.count('0'))
    # now we start to cross validate. how to deal with 0?
    pass


def train():
    pass


def predict():
    pass


if __name__ == '__main__':
    # check_vdjdb_slim()
    cross_validation()
