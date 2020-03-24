import pandas as pd
from Evaluations import load_model
import torch
from torch.utils.data import Dataset, DataLoader
from Loader import SignedPairsDataset
import collections
import pickle
from Sampler import get_diabetes_peptides
from argparse import Namespace

# todo distributions of diabetes peptides
# diabetes positive TCRS, NA, random
# peptides from mcpas and 4 from karen


class DiabetesTestDataset(SignedPairsDataset):
    def __init__(self, datafile, peptide):
        super().__init__(None)
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        all_pairs = []
        def invalid(seq):
            return pd.isna(seq) or any([aa not in self.amino_acids for aa in seq])
        data = pd.read_csv(datafile, engine='python')
        # first read all TRB, then unite with TRA according to sample id
        paired = {}
        for index in range(len(data)):
            id = int(data['libid'][index][3:])
            type = data['chain'][index]
            tcr = data['junction'][index]
            if type == 'b':
                tcrb = tcr
                v = ''
                j = ''
                protein = ''
                mhc = ''
                if invalid(tcrb):
                    paired[id] = None
                    continue
                tcr_data = ['UNK', tcrb, v, j]
                pep_data = (peptide, mhc, protein)
                try:
                    # alpha pair has already been seen
                    tcr_data, pep_data = paired[id]
                    tcr_data = list(tcr_data)
                    tcr_data[1] = tcrb
                    paired[id] = (tuple(tcr_data), pep_data)
                except KeyError:
                    paired[id] = (tuple(tcr_data), pep_data)
            if type == 'a':
                tcra = tcr
                tcr_data = [tcra, '', '', '']
                pep_data = (peptide, '', '')
                if invalid(tcra):
                    tcra = 'UNK'
                try:
                    # beta pair has already been seen
                    if paired[id] is None:
                        continue
                    tcr_data, pep_data = paired[id]
                    tcr_data = list(tcr_data)
                    tcr_data[0] = tcra
                    paired[id] = (tuple(tcr_data), pep_data)
                except KeyError:
                    paired[id] = (tuple(tcr_data), pep_data)
        # valid tcrb
        all_pairs.extend([t for t in list(paired.values()) if t is not None and t[0][1]])
        # train_pairs, test_pairs = train_test_split(set(all_pairs))
        # all is test at this point
        # todo we can add sign here, as what is positive in the csv column (some test)
        self.data = list(set(all_pairs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        def convert(seq):
            if seq == 'UNK':
                seq = [0]
            else:
                seq = [self.atox[aa] for aa in seq]
            return seq
        tcr_data, pep_data = self.data[index]
        # later we can fix this
        sign = 0
        tcra, tcrb, v, j = tcr_data
        peptide, mhc, protein = pep_data
        len_a = len(tcra) if tcra != 'UNK' else 0
        len_b = len(tcrb)
        len_p = len(peptide)
        tcra = convert(tcra)
        tcrb = convert(tcrb)
        peptide = convert(peptide)
        if sign == 1:
            weight = 5
        else:
            weight = 1
        sample = (tcra, len_a, tcrb, len_b, peptide, len_p, float(sign), float(weight))
        return sample
    pass


def check():
    datafile = 'diabetes_data/diabetes_tcrs.csv'
    test_dataset = DiabetesTestDataset(datafile, peptide='VEALYLVCG')
    dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4,
                            collate_fn=test_dataset.lstm_collate)
    for batch in dataloader:
        print(batch)
        # tcr length (including X, 0 for missing)
        # print(torch.sum(batch[0][0]).item())
        # tcra, tcrb, peps, pep_lens, sign, weight = batch
        # len_a = torch.sum(tcra, dim=[1, 2])
        # missing = (len_a == 0).nonzero(as_tuple=True)
        # print(missing)
        # full = len_a.nonzero(as_tuple=True)
        # print(full)
        # tcra_batch_ful = (tcra[full],)
        # tcrb_batch_ful = (tcrb[full],)
        # tcrb_batch_mis = (tcrb[missing],)
        # tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
        # tcr_batch_mis = (None, tcrb_batch_mis)
        exit()
    pass


def high_score_tcrs(peptide):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    xtoa = {index: amino for index, amino in enumerate(['PAD'] + amino_acids + ['X'])}
    def decode(seq):
        str = ''
        for i in range(len(seq)):
            if seq[i].item() == 0:
                break
            str += xtoa[seq[i].item()]
        return str
    datafile = 'diabetes_data/diabetes_tcrs.csv'
    # test_dataset = DiabetesTestDataset(datafile, peptide='VEALYLVCG')
    test_dataset = DiabetesTestDataset(datafile, peptide=peptide)
    if model.tcr_encoding_model == 'AE':
        collate_fn = test_dataset.ae_collate
    elif model.tcr_encoding_model == 'LSTM':
        collate_fn = test_dataset.lstm_collate
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=10, collate_fn=collate_fn)
    # outputs = []
    for batch_idx, batch in enumerate(loader):
        output = model.validation_step(batch, batch_idx)
        scores = output['y_hat']
        indicies = (scores > 0.95).nonzero(as_tuple=True)[0]
        if len(indicies):
            tcra, len_a, tcrb, len_b, peps, pep_lens, sign, weight = batch
            # print(tcrb[indicies])
            for tcr in tcrb[indicies]:
                print(decode(tcr))
            # positive = batch[indicies]
            # print(positive)
        # outputs.append(model.validation_step(batch, batch_idx))
    pass
    pass


def get_ergo_scores(samples, peptide):
    pass


def plot_distribution(peptide):
    pass


if __name__ == '__main__':
    # chack diabetes with different weight factor
    # checkpoint_path = 'mcpas_without_alpha/version_8/checkpoints/_ckpt_epoch_35.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_5/checkpoints/_ckpt_epoch_40.ckpt'
    checkpoint_path = 'mcpas_without_alpha/version_10/checkpoints/_ckpt_epoch_46.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_20/checkpoints/_ckpt_epoch_63.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_21/checkpoints/_ckpt_epoch_31.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_50/checkpoints/_ckpt_epoch_19.ckpt'
    # with alpha
    # checkpoint_path = 'mcpas_with_alpha/version_2/checkpoints/_ckpt_epoch_31.ckpt'
    args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
            'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    # check()
    for pep in ['VEALYLVCG', 'SRLGLWVRME', 'VLFGLGFAI', 'KRGIVEQCCTSISSL',
                'NFIRMVISNPAAT', 'KRGIVEQSSTSISSL', 'VELGGGPGA']:
        print('pep:', pep)
        high_score_tcrs(pep)
        print()

