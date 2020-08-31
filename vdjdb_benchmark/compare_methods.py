from netTCR_epitopes import EpitopeSlimNetTCR
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def roc(file, title):
    df = pd.read_csv(file)
    i = 0
    epitopes = list(set(df.Antigen.values))
    # creating subplots for each epitope. They will be plotted in one line
    fig, axes = plt.subplots(1, len(epitopes), figsize=(4 * len(epitopes), 2 * len(epitopes)))
    # creating a title
    plt.suptitle(f'Using data with duplicates, TRB, ' + title)

    # for each epitope
    for epitope, data in df.groupby('Antigen'):
        ax = axes[i]
        # predicted rank = 1 - probability of binding
        rank = data.prediction
        fpr, tpr, _ = roc_curve(data.label, rank)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})\nPositive = {len(data.loc[data.label == 1])}\n'
                                f'Negative = {len(data.loc[data.label == 0])}',
                c='darkorange')
        ax.set_title(epitope)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.legend(loc='lower right')
        i += 1
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    pass


if __name__ == '__main__':
    epitopes = ['GILGFVFTL', 'NLVPMVATV', 'ELAGIGILTV']
    chain = 'TRB'
    # netTCR = EpitopeSlimNetTCR(epitopes=epitopes, chain=chain, folder='.', duplicate=True,
    #                            prediction_path='nettcr_predictions.csv', predict=False)
    # netTCR.roc()

    # roc('nettcr_predictions.csv')
    roc('ergo_ae_mcpas.csv', 'Trained on McPAS')
    roc('ergo_ae_vdjdb.csv', 'Trained on VDJdb')
    roc('ergo_lstm_mcpas.csv', 'Trained on McPAS')
    roc('ergo_lstm_vdjdb.csv', 'Trained on VDJdb')

    pass
