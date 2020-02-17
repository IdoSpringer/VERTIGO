import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pickle
from Sampler import get_diabetes_peptides


class SignedPairsDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        def convert(seq):
            if seq == 'UNK':
                seq = [0]
            else:
                seq = [self.atox[aa] for aa in seq]
            return seq
        tcr_data, pep_data, sign = self.data[index]
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


    @staticmethod
    def get_max_length(x):
        return len(max(x, key=len))

    def pad_sequence(self, seq):
        def _pad(_it, _max_len):
            return _it + [0] * (_max_len - len(_it))
        return [_pad(it, self.get_max_length(seq)) for it in seq]

    def one_hot_encoding(self, tcr, max_len=28):
        tcr_batch = list(tcr)
        padding = torch.zeros(len(tcr_batch), max_len, 20 + 1)
        # TCR is converted to numbers at this point
        # We need to match the autoencoder atox, therefore -1
        for i in range(len(tcr_batch)):
            tcr_batch[i] = tcr_batch[i] + [self.atox['X']]
            for j in range(len(tcr_batch[i])):
                padding[i, j, tcr_batch[i][j] - 1] = 1
        return padding

    def ae_collate(self, batch):
        tcra, len_a, tcrb, len_b, peptide, len_p, sign, weight = zip(*batch)
        lst = []
        lst.append(torch.LongTensor(self.pad_sequence(tcra)))
        lst.append(torch.FloatTensor(self.one_hot_encoding(tcrb)))
        lst.append(torch.LongTensor(self.pad_sequence(peptide)))
        lst.append(torch.LongTensor(len_p))
        lst.append(torch.FloatTensor(sign))
        lst.append(torch.FloatTensor(weight))
        return lst

    def lstm_collate(self, batch):
        transposed = zip(*batch)
        lst = []
        for samples in transposed:
            if isinstance(samples[0], int):
                lst.append(torch.LongTensor(samples))
            elif isinstance(samples[0], float):
                lst.append(torch.FloatTensor(samples))
            elif isinstance(samples[0], collections.Sequence):
                lst.append(torch.LongTensor(self.pad_sequence(samples)))
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


def check():
    with open('mcpas_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    # train_dataset = SignedPairsDataset(train)
    train_dataset = DiabetesDataset(train, weight_factor=10)
    dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4,
                            collate_fn=train_dataset.lstm_collate)
    for batch in dataloader:
        print(batch)
        exit()
