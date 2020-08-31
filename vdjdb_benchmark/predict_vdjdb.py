# After some consideration we think that the optimal choice would be to compare
# 10-fold cross-validation results for each epitope against other epitopes.
# We would prefer to have raw performance values: TP, FP, TN, FN rates.
# We've also talked to TCRex team and they agreed to run this kind of benchmark once every 2-6 months
# for the most recent VDJdb update.
# There will be huge difference between these kind of methods and "unsupervised" ones
# which we'll mention in our benchmark description.
# We think to provide an additional score based on epitope assignments for 10^6 random TCR sequences
# that can tell about potential biases/overfitting, we'll get back to this later.
import pickle
import numpy as np
import pandas as pd
from random import shuffle
import Sampler
import Trainer
from argparse import ArgumentParser, Namespace
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


def read_file():
    datafile = 'VDJdb/vdjdb.txt'
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    data = pd.read_csv(datafile, engine='python', sep='\t')
    # complex id = 0 means missing alpha or beta
    # cpmplex id is not 0 meaning we need to pair both alpha and beta
    paired = {}
    for index in range(len(data)):
        sample = {}
        id = int(data['complex.id'][index])
        type = data['gene'][index]
        tcr = data['cdr3'][index]
        if type == 'TRA':
            tcra = tcr
            if id == 0:
                continue
            if invalid(tcra):
                tcra = 'UNK'
            sample['va'] = data['v.segm'][index]
            sample['ja'] = data['j.segm'][index]
            sample['tcra'] = tcra
            paired[id] = sample
        if type == 'TRB':
            if not id == 0:
                sample = paired[id]
                sample['tcrb'] = tcr
                sample['vb'] = data['v.segm'][index]
                sample['jb'] = data['j.segm'][index]
                sample['peptide'] = data['antigen.epitope'][index]
                sample['protein'] = data['antigen.gene'][index]
                sample['mhc'] = data['mhc.a'][index]
                sample['t_cell_type'] = data['mhc.class'][index]
                paired[id] = sample
            else:
                sample['tcrb'] = tcr
                sample['tcra'] = 'UNK'
                sample['va'] = 'UNK'
                sample['ja'] = 'UNK'
                sample['vb'] = data['v.segm'][index]
                sample['jb'] = data['j.segm'][index]
                sample['peptide'] = data['antigen.epitope'][index]
                sample['protein'] = data['antigen.gene'][index]
                sample['mhc'] = data['mhc.a'][index]
                sample['t_cell_type'] = data['mhc.class'][index]
            if invalid(tcr) or invalid(sample['peptide']):
                continue
            all_pairs.append(sample)
    return all_pairs


def train_test_split(pairs):
    shuffle(pairs)
    train_pairs = []
    test_pairs = []
    for pair in pairs:
        # 80% train, 20% test
        p = np.random.binomial(1, 0.8)
        if p == 1:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
    return train_pairs, test_pairs


def split_data_to_folds(all_pairs, k_fold=10):
    shuffle(all_pairs)
    jump = len(all_pairs) // k_fold
    print(len(all_pairs))
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    train_folds = []
    test_folds = []
    for i, fold in enumerate(chunks(all_pairs, jump)):
        train_fold, test_fold = train_test_split(fold)
        train_folds.append(train_fold)
        test_folds.append(test_fold)
    return train_folds, test_folds


def sample_train_and_test_folds(all_pairs, train_folds, test_folds):
    # save 10 train and 10 test pickles
    for i, (train_fold, test_fold) in enumerate(zip(train_folds, test_folds)):
        train_fold_pos = Sampler.positive_examples(train_fold)
        test_fold_pos = Sampler.positive_examples(test_fold)
        train_fold_neg = Sampler.negative_examples(train_fold, all_pairs, 5 * len(train_fold_pos))
        test_fold_neg = Sampler.negative_examples(test_fold, all_pairs, 5 * len(test_fold_pos))
        train = train_fold_pos + train_fold_neg
        shuffle(train)
        test = test_fold_pos + test_fold_neg
        shuffle(test)
        train_file = 'VDJdb/cross_validation_sets/vdjdb_train_fold' + str(i)
        test_file = 'VDJdb/cross_validation_sets/vdjdb_test_fold' + str(i)
        with open(str(train_file) + '.pickle', 'wb') as handle:
            pickle.dump(train, handle)
        with open(str(test_file) + '.pickle', 'wb') as handle:
            pickle.dump(test, handle)
    pass


def train(fold):
    parser = ArgumentParser()
    parser.add_argument('iter', type=int)
    parser.add_argument('gpu', type=int)
    parser.add_argument('dataset', type=str, default='vdjdb_fold')
    parser.add_argument('tcr_encoding_model', type=str, help='LSTM or AE')
    parser.add_argument('--cat_encoding', type=str, default='embedding')
    parser.add_argument('--use_alpha', action='store_true')
    parser.add_argument('--use_vj', action='store_true')
    parser.add_argument('--use_mhc', action='store_true')
    parser.add_argument('--use_t_type', action='store_true')
    parser.add_argument('--aa_embedding_dim', type=int, default=10)
    parser.add_argument('--cat_embedding_dim', type=int, default=50)
    parser.add_argument('--lstm_dim', type=int, default=500)
    parser.add_argument('--encoding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    number_of_gpu = torch.cuda.device_count()
    args = {'iter': fold, 'gpu': fold % number_of_gpu,
            'dataset': 'vdjdb_fold', 'tcr_encoding_model': 'AE', 'cat_encoding': 'embedding',
            'use_alpha': False, 'use_vj': True, 'use_mhc': False, 'use_t_type': False,
            'aa_embedding_dim': 10, 'cat_embedding_dim': 50,
            'lstm_dim': 500, 'encoding_dim': 100,
            'lr': 1e-3, 'wd': 1e-5,
            'dropout': 0.1}
    hparams = Namespace(**args)
    model = Trainer.ERGOLightning(hparams)
    # version flags
    version = ''
    version += str(hparams.iter)
    if hparams.dataset == 'vdjdb_fold':
        version += 'f'
    if hparams.tcr_encoding_model == 'AE':
        version += 'e'
    elif hparams.tcr_encoding_model == 'LSTM':
        version += 'l'
    if hparams.use_alpha:
        version += 'a'
    if hparams.use_vj:
        version += 'j'
    if hparams.use_mhc:
        version += 'h'
    if hparams.use_t_type:
        version += 't'
    logger = TensorBoardLogger("VDJdb/vdjdb_cv_logs", name="VDJdb/vdjdb_cv_models", version=version)
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
    trainer = Trainer(gpus=[hparams.gpu], logger=logger, early_stop_callback=early_stop_callback)
    trainer.fit(model)


def predict_multi_peptides():
    pass


def compute_score_metrics():
    # confusion matrix, TP FP TN FN
    pass


if __name__ == '__main__':
    all_pairs = read_file()
    train_folds, test_folds = split_data_to_folds(all_pairs)
    sample_train_and_test_folds(all_pairs, train_folds, test_folds)
    if sys.argv[1] == 'train':
        train(int(sys.argv[2]))
    pass
