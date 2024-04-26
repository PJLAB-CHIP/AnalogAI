import numpy as np
import torch
import os
from os.path import join,exists


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accuracy_min = np.Inf

    def __call__(self, 
                 accuracy, 
                 model,
                 client_idx, 
                 model_name, 
                 epoch, 
                 save_path):

        score = accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(accuracy, model, epoch, client_idx, model_name, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(accuracy, model, epoch, client_idx, model_name, save_path)
            self.counter = 0
        return self.best_score

    def save_checkpoint(self, accuracy, model, epoch, client_idx, model_name, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose :
            print(
                f'Best accuracy increased ({self.accuracy_min:.6f} --> {accuracy:.6f}).  Saving model ...'
            )
        os.makedirs(save_path,exist_ok=True)
        torch.save(
            model,
            join(save_path,'{}_client_{}_round_{}_{:.6f}.pth.tar'.format(model_name,client_idx,epoch, accuracy))
        )
        self.accuracy_min = accuracy