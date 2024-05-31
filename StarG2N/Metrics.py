
import math
import torch
import numpy as np

def r2_test(y_true, y_pre):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    # y = torch.from_numpy(y)
    return 1 - torch.sum((y_true - y_pre) ** 2) / torch.sum((y_true - torch.mean(y_pre)) ** 2)


def explained_variance_test(y_true, y_pre):
    # y = torch.from_numpy(y)
    return 1 - torch.var(y_true - y_pre) / torch.var(y_true)


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
    
def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    rmse = math.sqrt(np.mean(np.nan_to_num(mask * mse)))
    return rmse


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()
    
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))