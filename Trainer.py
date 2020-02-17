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


class ERGOLightning(pl.LightningModule):

    def __init__(self, dataset, tcr_encoding_model, use_alpha, embedding_dim, lstm_dim, encoding_dim, dropout=0.1):
        super(ERGOLightning, self).__init__()
        self.dataset = dataset
        # Model Type
        self.tcr_encoding_model = tcr_encoding_model
        self.use_alpha = use_alpha
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.encoding_dim = encoding_dim
        self.dropout = dropout
        # TCR Encoder
        if self.tcr_encoding_model == 'AE':
            self.tcrb_encoder = AE_Encoder(encoding_dim=encoding_dim)
        elif self.tcr_encoding_model == 'LSTM':
            if self.use_alpha:
                self.tcra_encoder = LSTM_Encoder(embedding_dim, lstm_dim, dropout)
            self.tcrb_encoder = LSTM_Encoder(embedding_dim, lstm_dim, dropout)
            self.encoding_dim = lstm_dim
        # Peptide Encoder
        self.pep_encoder = LSTM_Encoder(embedding_dim, lstm_dim, dropout)
        # MLP I (without alpha)
        self.mlp_dim1 = self.lstm_dim + self.encoding_dim
        self.hidden_layer1 = nn.Linear(self.mlp_dim1, int(np.sqrt(self.mlp_dim1)))
        self.relu = torch.nn.LeakyReLU()
        self.output_layer1 = nn.Linear(int(np.sqrt(self.mlp_dim1)), 1)
        self.dropout = nn.Dropout(p=dropout)
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

    def training_step(self, batch, batch_idx):
        # REQUIRED
        # could we save the batch as a dictionary? easy to unpack? more convenient to forward
        if self.tcr_encoding_model == 'AE':
            tcra, tcrb, peps, pep_lens, sign, weight = batch
            tcr_batch = (tcrb,)
        elif self.tcr_encoding_model == 'LSTM':
            tcra, len_a, tcrb, len_b, peps, pep_lens, sign, weight = batch
            # Detect samples with missing alpha
            if self.use_alpha:
                missing = (len_a == 0).nonzero(as_tuple=True)
                full = len_a.nonzero(as_tuple=True)
                tcra_batch_ful = (tcra[full], len_a[full])
                tcrb_batch_ful = (tcrb[full], len_b[full])
                tcrb_batch_mis = (tcrb[missing], len_b[missing])
                tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
                tcr_batch_mis = (None, tcrb_batch_mis)
                device = len_a.device
                y_hat = torch.zeros(len(sign)).to(device)
                y_hat_mis = self.forward(tcr_batch_mis, peps[missing], pep_lens[missing]).squeeze()
                y_hat[missing] = y_hat_mis
                if len(full) != 0:
                    y_hat_ful = self.forward(tcr_batch_ful, peps[full], pep_lens[full]).squeeze()
                    y_hat[full] = y_hat_ful
            else:
                tcrb_batch = (None, (tcrb, len_b))
                y_hat = self.forward(tcrb_batch, peps, pep_lens).squeeze()
        y = sign
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        if self.tcr_encoding_model == 'AE':
            tcra, tcrb, peps, pep_lens, sign, weight = batch
            tcr_batch = (tcrb,)
        elif self.tcr_encoding_model == 'LSTM':
            tcra, len_a, tcrb, len_b, peps, pep_lens, sign, weight = batch
            # Detect samples with missing alpha
            if self.use_alpha:
                missing = (len_a == 0).nonzero(as_tuple=True)
                full = len_a.nonzero(as_tuple=True)
                tcra_batch_ful = (tcra[full], len_a[full])
                tcrb_batch_ful = (tcrb[full], len_b[full])
                tcrb_batch_mis = (tcrb[missing], len_b[missing])
                tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
                tcr_batch_mis = (None, tcrb_batch_mis)
                device = len_a.device
                y_hat = torch.zeros(len(sign)).to(device)
                y_hat_mis = self.forward(tcr_batch_mis, peps[missing], pep_lens[missing]).squeeze()
                y_hat[missing] = y_hat_mis
                if len(full) != 0:
                    try:
                        y_hat_ful = self.forward(tcr_batch_ful, peps[full], pep_lens[full]).squeeze()
                        y_hat[full] = y_hat_ful
                    except RuntimeError:
                        print(full)
                        print(tcra)
                        print(len_a)
                        print(tcrb)
                        print(len_b)
            else:
                tcrb_batch = (None, (tcrb, len_b))
                y_hat = self.forward(tcrb_batch, peps, pep_lens).squeeze()
        y = sign
        return {'val_loss': F.binary_cross_entropy(y_hat, y), 'y_hat': y_hat, 'y': y}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        auc = roc_auc_score(y.cpu(), y_hat.cpu())
        print(auc)
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc}
        return {'avg_val_loss': avg_loss, 'val_auc': auc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        pass
    
    def test_end(self, outputs):
        # OPTIONAL
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
        return DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=10, collate_fn=collate_fn)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        pass

# todo plot regular ERGO ROC vs weighted ROC per diabetes weight factor


# model = ERGOLightning(tcr_encoding_model='AE', embedding_dim=10, lstm_dim=500, encoding_dim=100)
# trainer = Trainer(gpus=4, distributed_backend="ddp")

# model = ERGOLightning(dataset='mcpas', tcr_encoding_model='LSTM', use_alpha=False,
#                       embedding_dim=10, lstm_dim=500, encoding_dim=None)
# logger = TensorBoardLogger("lstm_alpha_logs", name="mcpas_without_alpha")
model = ERGOLightning(dataset='vdjdb', tcr_encoding_model='LSTM', use_alpha=True,
                      embedding_dim=10, lstm_dim=500, encoding_dim=None)
logger = TensorBoardLogger("lstm_alpha_logs", name="vdjdb_with_alpha")
early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
trainer = Trainer(gpus=[2], logger=logger, early_stop_callback=early_stop_callback)
trainer.fit(model)


# NOTE: fix sklearn import problem with this in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/
