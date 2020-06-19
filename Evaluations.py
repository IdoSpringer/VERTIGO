import pandas as pd
import numpy as np
import random
import pickle
from Loader import SignedPairsDataset, SinglePeptideDataset, get_index_dicts
from Trainer import ERGOLightning, ERGODiabetes
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace
from argparse import ArgumentParser
import torch
import Sampler
import csv
import pandas as pd
#  TPP-I
#  TPP-II
#  TPP-III
# SPB       V
#  Protein SPB
#  MPS
# all test today
# then we could check a trained model and compare tests to first ERGO paper


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


def load_model(hparams, checkpoint_path, diabetes=False):
    # args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
    #         'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
    # hparams = Namespace(**args)
    # model = ERGOLightning(hparams)
    # model.load_from_checkpoint('checkpoint_trial/version_4/checkpoints/_ckpt_epoch_27.ckpt')
    if diabetes:
        model = ERGODiabetes(hparams)
    else:
        model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_from_checkpoint('checkpoint')
    model.eval()
    return model


def spb(model, datafiles, peptide):
    test = get_tpp_ii_pairs(datafiles)
    test_dataset = SinglePeptideDataset(test, peptide)
    if model.tcr_encoding_model == 'AE':
        collate_fn = test_dataset.ae_collate
    elif model.tcr_encoding_model == 'LSTM':
        collate_fn = test_dataset.lstm_collate
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=10, collate_fn=collate_fn)
    outputs = []
    for batch_idx, batch in enumerate(loader):
        outputs.append(model.validation_step(batch, batch_idx))
    auc = model.validation_end(outputs)['val_auc']
    print(auc)
    pass


def spb_with_more_negatives(model, datafiles, peptide):
    test = get_tpp_ii_pairs(datafiles)
    # Regular SPB
    # test_dataset = SinglePeptideDataset(test, peptide)
    # More negatives
    test_dataset = SinglePeptideDataset(test, peptide, force_peptide=True, spb_force=True)
    if model.tcr_encoding_model == 'AE':
        collate_fn = test_dataset.ae_collate
    elif model.tcr_encoding_model == 'LSTM':
        collate_fn = test_dataset.lstm_collate
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=10, collate_fn=collate_fn)
    outputs = []
    i = 0
    positives = 0
    for batch_idx, batch in enumerate(loader):
        i += 1
        outputs.append(model.validation_step(batch, batch_idx))
    if i:
        print('positives:', int(torch.cat([x['y'] for x in outputs]).sum().item()))
        auc = model.validation_end(outputs)['val_auc']
        print(auc)
    pass


def multi_peptide_score(args, model, test_data, new_tcrs, number_of_peps):
    # take only positives from test with new TCRs
    tcrs = [p[0] for p in test_data if p[0] in new_tcrs and p[2] == 'p']
    targets = [p[1][0] for p in test_data if p[0] in new_tcrs and p[2] == 'p']
    # get N most frequent peps from the positives list
    peps = targets
    most_freq = []
    for i in range(number_of_peps):
        # find current most frequent pep
        freq_pep = max(peps, key=peps.count)
        most_freq.append(freq_pep)
        # remove all its instances from list
        peps = list(filter(lambda pep: pep != freq_pep, peps))
    # print(most_freq)
    score_matrix = np.zeros((len(tcrs), number_of_peps))
    for i in range(number_of_peps):
        try:
            # predict all new test TCRs with peps 1...k
            tcrs, _, scores = predict(args, model, tcrs, [most_freq[i]] * len(tcrs))
            score_matrix[:, i] = scores
        except ValueError:
            pass
        except IndexError:
            pass
        except TypeError:
            pass
    # true peptide targets indexes
    true_pred = list(map(lambda pep: most_freq.index(pep) if pep in most_freq else number_of_peps + 1, targets))
    accs = []
    for i in range(2, number_of_peps + 1):
        # define target pep using score argmax (save scores in a matrix)
        preds = np.argmax(score_matrix[:, :i], axis=1)
        # get accuracy score of k-class classification
        indices = [j for j in range(len(true_pred)) if true_pred[j] < i]
        k_class_predtion = np.array([preds[j] for j in indices])
        k_class_target = np.array([true_pred[j] for j in indices])
        accuracy = sum(k_class_predtion == k_class_target) / len(k_class_predtion)
        # print(accuracy)
        accs.append(accuracy)
    return most_freq, accs


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


def diabetes_test_set(model):
    # 8 paired samples, 4 peptides
    # tcra, tcrb, pep
    data = [('CAATRTSGTYKYIF', 'CASSPWGAGGTDTQYF', 'IGRPp39'),
            ('CAVGAGYGGATNKLIF', 'CASSFRGGGNPYEQYF', 'GADp70'),
            ('CAERLYGNNRLAF', 'CASTLLWGGDSYEQYF', 'GADp15'),
            ('CAVNPNQAGTALIF', 'CASAPQEAQPQHF', 'IGRPp31'),
            ('CALSDYSGTSYGKLTF', 'CASSLIPYNEQFF', 'GADp15'),
            ('CAVEDLNQAGTALIF', 'CASSLALGQGNQQFF', 'IGRPp31'),
            ('CILRDTISNFGNEKLTF', 'CASSFGSSYYGYTF', 'IGRPp39'),
            ('CAGQTGANNLFF', 'CASSQEVGTVPNQPQHF', 'IGRPp31')]
    peptide_map = {'IGRPp39': 'QLYHFLQIPTHEEHLFYVLS',
                   'GADp70': 'KVNFFRMVISNPAATHQDID',
                   'GADp15': 'DVMNILLQYVVKSFDRSTKV',
                   'IGRPp31': 'KWCANPDWIHIDTTPFAGLV'}
    true_labels = np.array([list(peptide_map.keys()).index(d[-1]) for d in data])
    print(true_labels)
    samples = []
    for tcra, tcrb, pep in data:
        tcr_data = (tcra, tcrb, 'v', 'j')
        pep_data = (peptide_map[pep], 'mhc', 'protein')
        samples.append((tcr_data, pep_data, 1))
    preds = np.zeros((len(samples), len(peptide_map)))
    for pep_idx, pep in enumerate(peptide_map):
        # signs do not matter here, we do only forward pass
        dataset = SinglePeptideDataset(samples, peptide_map[pep], force_peptide=True)
        if model.tcr_encoding_model == 'AE':
            collate_fn = dataset.ae_collate
        elif model.tcr_encoding_model == 'LSTM':
            collate_fn = dataset.lstm_collate
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=10, collate_fn=collate_fn)
        outputs = []
        for batch_idx, batch in enumerate(loader):
            outputs.append(model.validation_step(batch, batch_idx))
        y_hat = torch.cat([x['y_hat'].detach().cpu() for x in outputs])
        preds[:, pep_idx] = y_hat
    # print(preds)
    argmax = np.argmax(preds, axis=1)
    print(argmax)
    accuracy = sum((argmax == true_labels).astype(int)) / len(samples)
    print(accuracy)
    # try protein accuracy - IGRP and GAD
    true_labels = np.array([0 if x == 3 else 1 if x == 2 else x for x in true_labels])
    argmax = np.array([0 if x == 3 else 1 if x == 2 else x for x in argmax])
    print(true_labels)
    print(argmax)
    accuracy = sum((argmax == true_labels).astype(int)) / len(samples)
    print(accuracy)
    pass


def read_known_specificity_test(testfile):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    data = pd.read_csv(testfile, engine='python')
    for index in range(len(data)):
        sample = {}
        sample['tcra'] = data['junction_alpha'][index]
        sample['tcrb'] = data['junction_beta'][index]
        sample['va'] = data['v_gene_alpha'][index]
        sample['ja'] = data['j_gene_alpha'][index]
        sample['vb'] = data['v_gene_beta'][index]
        sample['jb'] = data['j_gene_beta'][index]
        sample['t_cell_type'] = 'UNK'
        sample['peptide'] = 'pep'
        sample['protein'] = 'protein'
        sample['mhc'] = 'UNK'
        if invalid(sample['tcrb']):
            continue
        if invalid(sample['tcra']):
            sample['tcra'] = 'UNK'
        # we do not use sign and weight, but it has to be defined
        sample['sign'] = 0
        sample['weight'] = 1
        all_pairs.append(sample)
    return all_pairs


# blind test (known specificity)
def diabetes_mps(hparams, model, testfile, pep_pool):
    with open('mcpas_human_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    if pep_pool == 4:
        peptide_map = {'IGRPp39': 'QLYHFLQIPTHEEHLFYVLS',
                       'GADp70': 'KVNFFRMVISNPAATHQDID',
                       'GADp15': 'DVMNILLQYVVKSFDRSTKV',
                       'IGRPp31': 'KWCANPDWIHIDTTPFAGLV'}
    else:
        peptide_map = {}
        with open(pep_pool, 'r') as file:
            file.readline()
            for line in file:
                pep, index, protein = line.strip().split(',')
                if protein in ['GAD', 'IGRP', 'Insulin']:
                    protein += 'p'
                pep_name = protein + index
                peptide_map[pep_name] = pep
    samples = read_known_specificity_test(testfile)
    preds = np.zeros((len(samples), len(peptide_map)))
    key_order = []
    for pep_idx, pep in enumerate(peptide_map):
        key_order.append(pep)
        testset = SinglePeptideDataset(samples, train_dicts, peptide_map[pep],
                                       force_peptide=True, spb_force=False)
        loader = DataLoader(testset, batch_size=10, shuffle=False, num_workers=10,
                            collate_fn=lambda b: testset.collate(b, tcr_encoding=hparams.tcr_encoding_model,
                                                                 cat_encoding=hparams.cat_encoding))
        outputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                model.eval()
                outputs.append(model.validation_step(batch, batch_idx))
        y_hat = torch.cat([x['y_hat'].detach().cpu() for x in outputs])
        preds[:, pep_idx] = y_hat
    argmax = np.argmax(preds, axis=1)
    predicted_peps = [key_order[i] for i in argmax]
    print(predicted_peps)
    pass


def mps():
    # today
    # try diabetes single cell (will probably fail but lets try)
    pass


def tpp_i():
    pass


def tpp_ii():
    pass


def tpp_iii():
    pass


if __name__ == '__main__':
    # chack diabetes with different weight factor
    # checkpoint_path = 'mcpas_without_alpha/version_8/checkpoints/_ckpt_epoch_35.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_5/checkpoints/_ckpt_epoch_40.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_10/checkpoints/_ckpt_epoch_46.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_20/checkpoints/_ckpt_epoch_63.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_21/checkpoints/_ckpt_epoch_31.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_50/checkpoints/_ckpt_epoch_19.ckpt'
    # with alpha
    # checkpoint_path = 'mcpas_with_alpha/version_2/checkpoints/_ckpt_epoch_31.ckpt'
    # with v, j , mhc
    # checkpoint_path = 'ergo_ii_diabetes/version_5/checkpoints/_ckpt_epoch_52.ckpt'
    # checkpoint_path = 'ergo_ii_diabetes/version_10/checkpoints/_ckpt_epoch_36.ckpt'
    checkpoint_path = 'ergo_ii_diabetes/version_20/checkpoints/_ckpt_epoch_40.ckpt'
    version = 20
    weight_factor = version
    args = {'version': version, 'gpu': 1,
            'dataset': 'mcpas_human', 'tcr_encoding_model': 'AE', 'cat_encoding': 'embedding',
            'use_alpha': True, 'use_vj': True, 'use_mhc': True,
            'aa_embedding_dim': 10, 'cat_embedding_dim': 50,
            'lstm_dim': 500, 'encoding_dim': 100,
            'lr': 1e-4, 'wd': 0,
            'dropout': 0.1,
            'weight_factor': weight_factor}

    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint, diabetes=True)
    # diabetes_test_set(model)
    diabetes_mps(hparams, model, 'diabetes_data/known_specificity.csv', 'diabetes_data/28pep_pool.csv')
    # diabetes_mps(hparams, model, 'diabetes_data/known_specificity.csv', pep_pool=4)

    # train_pickle = model.dataset + '_train_samples.pickle'
    # test_pickle = model.dataset + '_test_samples.pickle'
    # datafiles = train_pickle, test_pickle
    # spb(model, datafiles, peptide='LPRRSGAAGA')
    # spb(model, datafiles, peptide='GILGFVFTL')
    # spb(model, datafiles, peptide='NLVPMVATV')
    # spb(model, datafiles, peptide='GLCTLVAML')
    # spb(model, datafiles, peptide='SSYRRPVGI')
    # d_peps = list(Sampler.get_diabetes_peptides('data/McPAS-TCR.csv'))
    # for pep in d_peps:
    #     try:
    #         print(pep)
    #         spb_with_more_negatives(model, datafiles, peptide=pep)
    #     except ValueError:
    #         pass
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
