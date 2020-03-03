import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Loader import SignedPairsDataset, DiabetesDataset
from Models import PaddingAutoencoder, AE_Encoder, LSTM_Encoder, ERGO
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
# keep up the good work :)
# todo AE for alpha mechanism


class ERGOLightning(pl.LightningModule):

    def __init__(self, hparams):
        super(ERGOLightning, self).__init__()
        self.hparams = hparams
        self.dataset = hparams.dataset
        # Model Type
        self.tcr_encoding_model = hparams.tcr_encoding_model
        self.use_alpha = hparams.use_alpha
        # Dimensions
        self.embedding_dim = hparams.embedding_dim
        self.lstm_dim = hparams.lstm_dim
        self.encoding_dim = hparams.encoding_dim
        self.dropout = hparams.dropout
        # TCR Encoder
        if self.tcr_encoding_model == 'AE':
            if self.use_alpha:
                self.tcra_encoder = AE_Encoder(encoding_dim=self.encoding_dim, tcr_type='alpha')
            self.tcrb_encoder = AE_Encoder(encoding_dim=self.encoding_dim, tcr_type='beta')
        elif self.tcr_encoding_model == 'LSTM':
            if self.use_alpha:
                self.tcra_encoder = LSTM_Encoder(self.embedding_dim, self.lstm_dim, self.dropout)
            self.tcrb_encoder = LSTM_Encoder(self.embedding_dim, self.lstm_dim, self.dropout)
            self.encoding_dim = self.lstm_dim
        # Peptide Encoder
        self.pep_encoder = LSTM_Encoder(self.embedding_dim, self.lstm_dim, self.dropout)
        # MLP I (without alpha)
        self.mlp_dim1 = self.lstm_dim + self.encoding_dim
        self.hidden_layer1 = nn.Linear(self.mlp_dim1, int(np.sqrt(self.mlp_dim1)))
        self.relu = torch.nn.LeakyReLU()
        self.output_layer1 = nn.Linear(int(np.sqrt(self.mlp_dim1)), 1)
        self.dropout = nn.Dropout(p=self.dropout)
        if self.use_alpha:
            # MLP II (with alpha)
            self.mlp_dim2 = self.lstm_dim + 2 * self.encoding_dim
            self.hidden_layer2 = nn.Linear(self.mlp_dim2, int(np.sqrt(self.mlp_dim2)))
            self.output_layer2 = nn.Linear(int(np.sqrt(self.mlp_dim2)), 1)

    def forward(self, tcr_batch, peps, pep_lens):
        # PEPTIDE Encoder:
        pep_encoding = self.pep_encoder(peps, pep_lens)
        # TCR Encoder:
        tcra, tcrb = tcr_batch
        tcrb_encoding = self.tcrb_encoder(*tcrb)
        if tcra:
            tcra_encoding = self.tcra_encoder(*tcra)
            # MLP II Classifier
            tcr_pep_concat = torch.cat([tcra_encoding, tcrb_encoding, pep_encoding], 1)
            hidden_output = self.dropout(self.relu(self.hidden_layer2(tcr_pep_concat)))
            mlp_output = self.output_layer2(hidden_output)
        else:
            # MLP I Classifier
            tcr_pep_concat = torch.cat([tcrb_encoding, pep_encoding], 1)
            hidden_output = self.dropout(self.relu(self.hidden_layer1(tcr_pep_concat)))
            mlp_output = self.output_layer1(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output

    def step(self, batch):
        if self.use_alpha:
            if self.tcr_encoding_model == 'AE':
                tcra, tcrb, peps, pep_lens, sign, weight = batch
                len_a = torch.sum(tcra, dim=[1, 2])
                missing = (len_a == 0).nonzero(as_tuple=True)
                full = len_a.nonzero(as_tuple=True)
                tcra_batch_ful = (tcra[full],)
                tcrb_batch_ful = (tcrb[full],)
                tcrb_batch_mis = (tcrb[missing],)
                tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
                tcr_batch_mis = (None, tcrb_batch_mis)
            elif self.tcr_encoding_model == 'LSTM':
                tcra, len_a, tcrb, len_b, peps, pep_lens, sign, weight = batch
                missing = (len_a == 0).nonzero(as_tuple=True)
                full = len_a.nonzero(as_tuple=True)
                tcra_batch_ful = (tcra[full], len_a[full])
                tcrb_batch_ful = (tcrb[full], len_b[full])
                tcrb_batch_mis = (tcrb[missing], len_b[missing])
                tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
                tcr_batch_mis = (None, tcrb_batch_mis)
            device = len_a.device
            y_hat = torch.zeros(len(sign)).to(device)
            # there are samples without alpha
            if len(missing[0]):
                y_hat_mis = self.forward(tcr_batch_mis, peps[missing], pep_lens[missing]).squeeze()
                y_hat[missing] = y_hat_mis
            # there are samples with alpha
            if len(full[0]):
                y_hat_ful = self.forward(tcr_batch_ful, peps[full], pep_lens[full]).squeeze()
                y_hat[full] = y_hat_ful
        else:
            if self.tcr_encoding_model == 'AE':
                tcra, tcrb, peps, pep_lens, sign, weight = batch
                tcrb_batch = (None, (tcrb,))
                y_hat = self.forward(tcrb_batch, peps, pep_lens).squeeze()
            elif self.tcr_encoding_model == 'LSTM':
                tcra, len_a, tcrb, len_b, peps, pep_lens, sign, weight = batch
                tcrb_batch = (None, (tcrb, len_b))
                y_hat = self.forward(tcrb_batch, peps, pep_lens).squeeze()
        y = sign
        return y, y_hat

    def training_step(self, batch, batch_idx):
        # REQUIRED
        self.train()
        y, y_hat = self.step(batch)
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        self.eval()
        y, y_hat = self.step(batch)
        return {'val_loss': F.binary_cross_entropy(y_hat, y), 'y_hat': y_hat, 'y': y}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        # auc = roc_auc_score(y.cpu(), y_hat.cpu())
        auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        # print(auc)
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc}
        return {'avg_val_loss': avg_loss, 'val_auc': auc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        # result = self.validation_step(batch, batch_idx)
        # return {'y_hat': result['y_hat'], 'y': result['y']}
        pass
    
    def test_end(self, outputs):
        # OPTIONAL
        # y = torch.cat([x['y'] for x in outputs])
        # y_hat = torch.cat([x['y_hat'] for x in outputs])
        # auc = roc_auc_score(y.cpu(), y_hat.cpu())
        # print(auc)
        # return {'spb_auc': auc}
        pass

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        with open(self.dataset + '_train_samples.pickle', 'rb') as handle:
            train = pickle.load(handle)
        train_dataset = SignedPairsDataset(train)
        if self.tcr_encoding_model == 'AE':
            collate_fn = train_dataset.ae_collate
        elif self.tcr_encoding_model == 'LSTM':
            collate_fn = train_dataset.lstm_collate
        return DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10, collate_fn=collate_fn)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        with open(self.dataset + '_test_samples.pickle', 'rb') as handle:
            test = pickle.load(handle)
        test_dataset = SignedPairsDataset(test)
        if self.tcr_encoding_model == 'AE':
            collate_fn = test_dataset.ae_collate
        elif self.tcr_encoding_model == 'LSTM':
            collate_fn = test_dataset.lstm_collate
        return DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=10, collate_fn=collate_fn)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        pass


class ERGODiabetes(ERGOLightning):

    def __init__(self, weight_factor, dataset, tcr_encoding_model, use_alpha,
                 embedding_dim, lstm_dim, encoding_dim, dropout=0.1):
        super().__init__(dataset, tcr_encoding_model, use_alpha, embedding_dim,
                         lstm_dim, encoding_dim, dropout=0.1)
        self.weight_factor = weight_factor

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        with open(self.dataset + '_train_samples.pickle', 'rb') as handle:
            train = pickle.load(handle)
        train_dataset = DiabetesDataset(train, weight_factor=self.weight_factor)
        if self.tcr_encoding_model == 'AE':
            collate_fn = train_dataset.ae_collate
        elif self.tcr_encoding_model == 'LSTM':
            collate_fn = train_dataset.lstm_collate
        return DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10, collate_fn=collate_fn)


def main():
    model = ERGODiabetes(dataset='mcpas', tcr_encoding_model='LSTM', use_alpha=False,
                          embedding_dim=10, lstm_dim=500, encoding_dim=None, weight_factor=20)
    logger = TensorBoardLogger("diabetes_logs", name="mcpas_without_alpha", version='20')
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
    trainer = Trainer(gpus=[2], logger=logger, early_stop_callback=early_stop_callback)
    trainer.fit(model)


if __name__ == '__main__':
    # main()
    pass


# NOTE: fix sklearn import problem with this in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/
