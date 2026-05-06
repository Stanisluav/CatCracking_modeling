import numpy as np
import torch
import torch.nn as nn

#==============================================================
# coefficient of determination
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

#==============================================================
# root mean square error
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

#==============================================================
# Function for validation (val loss, MSE, MAE)
def compute_metrics(y_pred, y_true):
    with torch.no_grad():
        loss_fn = nn.MSELoss()
        mse = loss_fn(y_pred, y_true)
        y_pred_pct = y_pred * 100.0
        y_true_pct = y_true * 100.0
        rmse_pct = rmse(y_true_pct.cpu().numpy(), y_pred_pct.cpu().numpy())
        mae_pct = torch.mean(torch.abs(y_pred_pct - y_true_pct))
    return mse.item(), rmse_pct.item(), mae_pct.item()