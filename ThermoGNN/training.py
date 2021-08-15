import random

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torchmetrics.functional import pearson_corrcoef


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    total_train_loss = 0
    train_data_size = 0
    for data in train_loader:
        data = data.to(device)
        out = model(data)
        if criterion.__repr__() == "WeightedMSELoss()":
            loss = criterion(out, data.y, data.wy)
        else:
            loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss * out.size(0)
        train_data_size += out.size(0)

    train_loss = total_train_loss / train_data_size

    model.eval()
    total_valid_loss = 0
    valid_data_size = 0
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            out = model(data)
            loss = mse_loss(out, data.y)
            total_valid_loss += loss * out.size(0)
            valid_data_size += out.size(0)

    valid_loss = total_valid_loss / valid_data_size

    return train_loss, valid_loss


def evaluate(model, loader, device, return_tensor=False):
    model.eval()
    pred = []
    y = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred.append(out)
            y.append(data.y)

        pred_tensor = torch.cat(pred)
        y_tensor = torch.cat(y)
        corr = pearson_corrcoef(pred_tensor, y_tensor)
        rmse = torch.sqrt(mse_loss(pred_tensor, y_tensor))

    if return_tensor:
        return pred_tensor, y_tensor
    else:
        return corr, rmse


def metrics(pred_dir, pred_rev, y_dir, y_rev):
    corr_dir = pearson_corrcoef(pred_dir, y_dir)
    rmse_dir = torch.sqrt(mse_loss(pred_dir, y_dir))
    corr_rev = pearson_corrcoef(pred_rev, y_rev)
    rmse_rev = torch.sqrt(mse_loss(pred_rev, y_rev))
    corr_dir_rev = pearson_corrcoef(pred_dir, pred_rev)
    delta = torch.mean(pred_dir + pred_rev)

    return corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta


class EarlyStopping:

    def __init__(self, patience=10, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, score, model, goal="maximize"):

        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), self.path)
        self.best_score = score
