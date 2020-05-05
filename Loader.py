import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pickle
from Sampler import get_diabetes_peptides
import pandas as pd
import math


class SignedPairsDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        self.all_va = [sample['va'] for sample in samples if not pd.isna(sample['va'])]
        self.vatox = {va: index for index, va in enumerate(set(self.all_va), 1)}
        self.vatox['UNK'] = 0
        self.all_vb = [sample['vb'] for sample in samples if not pd.isna(sample['vb'])]
        self.vbtox = {vb: index for index, vb in enumerate(set(self.all_vb), 1)}
        self.vbtox['UNK'] = 0
        self.all_ja = [sample['ja'] for sample in samples if not pd.isna(sample['ja'])]
        self.jatox = {ja: index for index, ja in enumerate(set(self.all_ja), 1)}
        self.jatox['UNK'] = 0
        self.all_jb = [sample['jb'] for sample in samples if not pd.isna(sample['jb'])]
        self.jbtox = {jb: index for index, jb in enumerate(set(self.all_jb), 1)}
        self.jbtox['UNK'] = 0
        self.all_mhc = [sample['mhc'] for sample in samples if not pd.isna(sample['mhc'])]
        self.mhctox = {mhc: index for index, mhc in enumerate(set(self.all_mhc), 1)}
        self.mhctox['UNK'] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

    def aa_convert(self, seq):
        if seq == 'UNK':
            seq = []
        else:
            seq = [self.atox[aa] for aa in seq]
        return seq

    @staticmethod
    def get_max_length(x):
        return len(max(x, key=len))

    def seq_letter_encoding(self, seq):
        def _pad(_it, _max_len):
            return _it + [0] * (_max_len - len(_it))
        return [_pad(it, self.get_max_length(seq)) for it in seq]

    def seq_one_hot_encoding(self, tcr, max_len=28):
        tcr_batch = list(tcr)
        padding = torch.zeros(len(tcr_batch), max_len, 20 + 1)
        # TCR is converted to numbers at this point
        # We need to match the autoencoder atox, therefore -1
        for i in range(len(tcr_batch)):
            # missing alpha
            if tcr_batch[i] == [0]:
                continue
            tcr_batch[i] = tcr_batch[i] + [self.atox['X']]
            for j in range(min(len(tcr_batch[i]), max_len)):
                padding[i, j, tcr_batch[i][j] - 1] = 1
        return padding

    def label_encoding(self):
        # this will be label with learned embedding matrix (so not 1 dimension)
        # get all possible tags
        # in init ?
        pass

    def binary_encoding(self):
        pass

    def hashing_encoding(self):
        # I think that feature hashing is not relevant in this case
        pass

    @staticmethod
    def binarize(num):
        l = []
        while num:
            l.append(num % 2)
            num //= 2
        l.reverse()
        # print(l)
        return l

    def collate(self, batch, tcr_encoding, cat_encoding):
        lst = []
        # TCRs
        tcrb = [self.aa_convert(sample['tcrb']) for sample in batch]
        tcra = [self.aa_convert(sample['tcra']) for sample in batch]
        if tcr_encoding == 'ae':
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcra, max_len=34)))
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcrb)))
        elif tcr_encoding == 'lstm':
            lst.append(torch.LongTensor(self.seq_letter_encoding(tcra)))
            len_a = [len(a) for a in tcra]
            lst.append(torch.LongTensor(len_a))
            lst.append(torch.LongTensor(self.seq_letter_encoding(tcrb)))
            len_b = [len(b) for b in tcrb]
            lst.append(torch.LongTensor(len_b))
        # Peptide
        peptide = [self.aa_convert(sample['peptide']) for sample in batch]
        lst.append(torch.LongTensor(self.seq_letter_encoding(peptide)))
        len_p = [len(p) for p in peptide]
        lst.append(torch.LongTensor(len_p))
        # Categorical features - V alpha, V beta, J alpha, J beta, MHC
        categorical = ['va', 'vb', 'ja', 'jb', 'mhc']
        cat_idx = [self.vatox, self.vbtox, self.jatox, self.jbtox, self.mhctox]
        for cat, idx in zip(categorical, cat_idx):
            batch_cat = ['UNK' if pd.isna(sample[cat]) else sample[cat] for sample in batch]
            batch_idx = list(map(lambda x: idx[x], batch_cat))
            # print(cat)
            if cat_encoding == 'embedding':
                # label encoding
                batch_cat = batch_idx
            if cat_encoding == 'binary':
                # we need a matrix for the batch with the binary encodings
                max_len = int(math.log(len(idx), 2)) + 1

                def bin_pad(num, _max_len):
                    bin_list = self.binarize(num)
                    return [0] * (_max_len - len(bin_list)) + bin_list
                bin_mat = torch.tensor([bin_pad(v, max_len) for v in batch_idx])
                batch_cat = bin_mat
            lst.append(batch_cat)
        # T cell type
        pass
        # Sign
        sign = [sample['sign'] for sample in batch]
        lst.append(torch.FloatTensor(sign))
        # weight will be handled in trainer (it is not loader job)
        # lst.append(torch.FloatTensor(weight))
        return lst

    pass


class DiabetesDataset(SignedPairsDataset):
    def __init__(self, samples, weight_factor):
        super().__init__(samples)
        self.diabetes_peptides = get_diabetes_peptides('data/McPAS-TCR.csv')
        self.weight_factor = weight_factor

    def __getitem__(self, index):
        sample = list(super().__getitem__(index))
        weight = sample[-1]
        tcr_data, pep_data, sign = self.data[index]
        peptide, mhc, protein = pep_data
        if peptide in self.diabetes_peptides:
            weight *= self.weight_factor
        return tuple(sample[:-1] + [weight])
    pass


class SinglePeptideDataset(SignedPairsDataset):
    def __init__(self, samples, peptide, force_peptide=False, spb_force=False):
        super().__init__(samples)
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        if force_peptide:
            if spb_force:
                pep_data = (peptide, 'mhc', 'protein')
                self.data = []
                for pair in samples:
                    if pair[1][0] != peptide:
                        self.data.append((pair[0], pep_data, 0))
                    # we keep the original positives
                    else:
                        self.data.append(pair)
            else:
                # we do it only for MPS and we have to check that the signs are correct
                pep_data = (peptide, 'mhc', 'protein')
                self.data = [(pair[0], pep_data, pair[-1]) for pair in samples]
        else:
            self.data = [pair for pair in samples if pair[1][0] == peptide]


def check():
    with open('mcpas_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    train_dataset = SignedPairsDataset(train)
    # train_dataset = DiabetesDataset(train, weight_factor=10)
    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=4,
                            collate_fn=lambda b: train_dataset.collate(b, tcr_encoding='lstm',
                                                                       cat_encoding='embedding'))
    for batch in dataloader:
        pass
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


check()
