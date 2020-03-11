import pandas as pd
# todo distributions of diabetes peptides
# diabetes positive TCRS, NA, random
# peptides from mcpas and 4 from karen


def read_diabetes_data(datafile):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    data = pd.read_csv(datafile, engine='python')
    # first read all TRB, then unite with TRA according to sample id
    paired = {}
    for index in range(len(data)):
        id = int(data['libid'][index])
        type = data['chain'][index]
        tcr = data['junction'][index]
        if type == 'b':
            tcrb = tcr
            v = ''
            j = ''
            peptide = ''
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
            pep_data = ('', '', '')
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

    train_pairs, test_pairs = train_test_split(set(all_pairs))
    return all_pairs, train_pairs, test_pairs
    # read csv
    # get samples with signs
    pass


def get_ergo_scores(samples, peptide):
    pass


def plot_distribution(peptide):
    pass
