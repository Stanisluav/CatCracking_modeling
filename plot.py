import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from dataset import load_experimental_data, denormalize_concentrations
from utils import r2_score, rmse

parser = argparse.ArgumentParser(description='Validate KCONDE model for FCC')
parser.add_argument('--model', type=str, default='flexible')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to model checkpoint (default: kconde_flexible.pth or kconde_strict.pth)')
parser.add_argument('--hidden', type=int, default=None,
                    help='Hidden layer size (must match the trained model)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == 'flexible':
    from neuralODE_model import KCONDE_FCC
    hidden_size = args.hidden if args.hidden is not None else 128
    model = KCONDE_FCC(hidden_size=hidden_size).to(device)
    checkpoint = args.checkpoint if args.checkpoint else 'kconde_flexible.pth'
    normalize_T = True
else:
    pass
    # Пока только одна модель

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

experiments = load_experimental_data()

C_O_list = [e['C_O'] for e in experiments]
t_s_list = [e['t_s'] for e in experiments]
T_C_list = [e['T_C'] for e in experiments]

targets = np.array([e['target'] for e in experiments])   # (N, 5)
C_O_t = torch.tensor(C_O_list, dtype=torch.float32).to(device) / 10.0
t_s_t = torch.tensor(t_s_list, dtype=torch.float32).to(device)

if normalize_T:
    T_t = (torch.tensor(T_C_list, dtype=torch.float32).to(device) - 500.0) / 100.0
else:
    T_t = torch.tensor([t + 273.15 for t in T_C_list], dtype=torch.float32).to(device)

C0 = torch.zeros(len(experiments), 5).to(device)
C0[:, 0] = 1.0

with torch.no_grad():
    pred_norm = model.integrate_batch(C0, t_s_t, C_O_t, T_t)
    pred = denormalize_concentrations(pred_norm.cpu().numpy())

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
plt.savefig(f'validation_{args.model}.png', dpi=300, bbox_inches='tight')
plt.show()