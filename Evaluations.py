import pandas as pd
import numpy as np
import random
import pickle
from Loader import SignedPairsDataset, SinglePeptideDataset
from Trainer import ERGOLightning
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace
import torch
import Sampler
# todo all tests suggested in ERGO
# todo TPP-I
# todo TPP-II
# todo TPP-III
# todo SPB
# todo Protein SPB
# todo MPS


def get_new_tcrs_and_peps(datafiles):
    train_pickle, test_pickle = datafiles
    # open and read data
    # return TCRs and peps that appear only in test pairs
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test = pickle.load(handle)
    train_peps = [t[1][0] for t in train]
    train_tcrbs = [t[0][1] for t in train]
    test_peps = [t[1][0] for t in test]
    test_tcrbs = [t[0][1] for t in test]
    new_test_tcrbs = set(test_tcrbs).difference(set(train_tcrbs))
    new_test_peps = set(test_peps).difference(set(train_peps))
    # print(len(set(test_tcrbs)), len(new_test_tcrbs))
    return new_test_tcrbs, new_test_peps


def get_tpp_ii_pairs(datafiles):
    # We only take new TCR beta chains (why? does it matter?)
    train_pickle, test_pickle = datafiles
    with open(test_pickle, 'rb') as handle:
        test_data = pickle.load(handle)
    new_test_tcrbs, new_test_peps = get_new_tcrs_and_peps(datafiles)
    return [t for t in test_data if t[0][1] in new_test_tcrbs]


def load_model(hparams, checkpoint_path):
    # args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
    #         'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
    # hparams = Namespace(**args)
    # model = ERGOLightning(hparams)
    # model.load_from_checkpoint('checkpoint_trial/version_4/checkpoints/_ckpt_epoch_27.ckpt')
    model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_from_checkpoint('checkpoint')
    return model


def spb(model, datafiles, peptide):
    test = get_tpp_ii_pairs(datafiles)
    test_dataset = SinglePeptideDataset(test, peptide)
    if model.tcr_encoding_model == 'AE':
        collate_fn = test_dataset.ae_collate
    elif model.tcr_encoding_model == 'LSTM':
        collate_fn = test_dataset.lstm_collate
    loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=10, collate_fn=collate_fn)
    outputs = []
    for batch_idx, batch in enumerate(loader):
        outputs.append(model.validation_step(batch, batch_idx))
    auc = model.validation_end(outputs)['val_auc']
    print(auc)
    pass

#
# def check():
#     args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
#             'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
#     hparams = Namespace(**args)
#     checkpoint = 'checkpoint_trial/version_4/checkpoints/_ckpt_epoch_27.ckpt'
#     model = load_model(hparams, checkpoint)
#     train_pickle = model.dataset + '_train_samples.pickle'
#     test_pickle = model.dataset + '_test_samples.pickle'
#     datafiles = train_pickle, test_pickle
#     # spb(model, datafiles, peptide='LPRRSGAAGA')
#     # spb(model, datafiles, peptide='GILGFVFTL')
#     # spb(model, datafiles, peptide='NLVPMVATV')
#     # spb(model, datafiles, peptide='GLCTLVAML')
#     # spb(model, datafiles, peptide='SSYRRPVGI')
#     d_peps = Sampler.get_diabetes_peptides('data/McPAS-TCR.csv')
#     print(d_peps)
#     pass


def check2(checkpoint_path):
    args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
            'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    train_pickle = model.dataset + '_train_samples.pickle'
    test_pickle = model.dataset + '_test_samples.pickle'
    datafiles = train_pickle, test_pickle
    spb(model, datafiles, peptide='LPRRSGAAGA')
    spb(model, datafiles, peptide='GILGFVFTL')
    spb(model, datafiles, peptide='NLVPMVATV')
    spb(model, datafiles, peptide='GLCTLVAML')
    spb(model, datafiles, peptide='SSYRRPVGI')
    d_peps = list(Sampler.get_diabetes_peptides('data/McPAS-TCR.csv'))
    print(d_peps)
    for pep in d_peps:
        try:
            print(pep)
            spb(model, datafiles, peptide=pep)
        except ValueError:
            pass

    pass

# chack diabetes with different weight factor
# checkpoint_path = 'mcpas_without_alpha/version_8/checkpoints/_ckpt_epoch_35.ckpt'
# checkpoint_path = 'mcpas_without_alpha/version_5/checkpoints/_ckpt_epoch_40.ckpt'
# checkpoint_path = 'mcpas_without_alpha/version_10/checkpoints/_ckpt_epoch_46.ckpt'
# checkpoint_path = 'mcpas_without_alpha/version_20/checkpoints/_ckpt_epoch_63.ckpt'
# with alpha
# checkpoint_path = 'mcpas_with_alpha/version_2/checkpoints/_ckpt_epoch_31.ckpt'
check2(checkpoint_path)

def mps():
    pass


def tpp_i():
    pass


def tpp_ii():
    pass


def tpp_iii():
    pass

# it should be easy because the datasets are fixed and the model is saved in a lightning checkpoint
# tests might be implemented in lightning module

#
# args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
#            'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
# hparams = Namespace(**args)
# model = ERGOLightning(hparams)
# logger = TensorBoardLogger("trial_logs", name="checkpoint_trial")
# early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
# trainer = Trainer(gpus=[2], logger=logger, early_stop_callback=early_stop_callback)
# trainer.fit(model)
