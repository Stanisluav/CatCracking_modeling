import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_experimental_data
from kconde_model import KCONDE_FCC

#==============================================================
# Расчет метрик
#==============================================================
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

#==============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KCONDE_FCC().to(device)
model.load_state_dict(torch.load('kconde_fast.pth', map_location=device))
model.eval()

experiments = load_experimental_data()

# подготовка данных
C_O_list = [e['C_O'] for e in experiments]
t_s_list = [e['t_s'] for e in experiments]
T_list = [e['T_C'] for e in experiments]
targets = np.array([e['target'] for e in experiments])   # (N, 5) в %

C_O_t = torch.tensor(C_O_list, dtype=torch.float32).to(device) / 10.0
t_s_t = torch.tensor(t_s_list, dtype=torch.float32).to(device)
T_t = (torch.tensor(T_list, dtype=torch.float32).to(device) - 500.0) / 100.0

C0 = torch.zeros(len(experiments), 5).to(device)
C0[:, 0] = 1.0

with torch.no_grad():
    pred_norm = model.integrate_batch(C0, t_s_t, C_O_t, T_t)
    pred = pred_norm.cpu().numpy() * 100.0


names = ['VGO', 'LCO', 'Gasoline', 'Light gases', 'Coke']
plt.figure(figsize=(12, 10))

for i, name in enumerate(names):
    y_true = targets[:, i]
    y_pred = pred[:, i]

    r2 = r2_score(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)

    plt.subplot(2, 3, i + 1)
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')
    plt.plot([0, 100], [0, 100], 'k--', linewidth=1)
    plt.xlabel('Experimental, %')
    plt.ylabel('Predicted, %')
    plt.title(f'{name}\nR² = {r2:.3f}   RMSE = {rmse_val:.2f} %')
    plt.grid(True)

plt.tight_layout()
plt.savefig('validation.png', dpi=300, bbox_inches='tight')
plt.show()