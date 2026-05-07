import torch
import torch.nn as nn
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from utils import compute_metrics
from dataset import load_experimental_data, normalize_concentrations, denormalize_concentrations

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='flexible')
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
args = parser.parse_args()


if args.model == 'flexible':
    from neuralODE_model import KCONDE_FCC
    model_class = KCONDE_FCC
    hidden_size = args.hidden if args.hidden is not None else 128
    normalize_T = True   # T нормируется ( (T-500)/100 )
else:
    pass
    # Возмлжно позже будут другие архитектуры

print(f"Training {args.model} model | hidden_size={hidden_size} | epochs={args.epochs} | lr={args.lr}")

experiments = load_experimental_data()
C_O_list = []
t_s_list = []
T_list_raw = []
targets_norm = []

for exp in experiments:
    C_O_list.append(exp['C_O'])
    t_s_list.append(exp['t_s'])
    T_list_raw.append(exp['T_C'])
    targets_norm.append(torch.tensor(normalize_concentrations(exp['target']), dtype=torch.float32))

C_O_batch = torch.tensor(C_O_list, dtype=torch.float32)
t_s_batch = torch.tensor(t_s_list, dtype=torch.float32)

if normalize_T:
    T_batch = (torch.tensor(T_list_raw, dtype=torch.float32) - 500.0) / 100.0
else:
    T_batch = torch.tensor([t + 273.15 for t in T_list_raw], dtype=torch.float32)

targets_batch = torch.stack(targets_norm)

# C/O normalization
C_O_batch = C_O_batch / 10.0   # C/O обычно 1..7

C0_batch = torch.zeros(len(experiments), 5)
C0_batch[:, 0] = 1.0

indices = np.arange(len(experiments))
train_idx, val_idx = train_test_split(indices, test_size=args.val_split)

def subset_tensors(*tensors):
    return [t[train_idx] for t in tensors], [t[val_idx] for t in tensors]

(train_data, val_data) = subset_tensors(
    C0_batch, C_O_batch, t_s_batch, T_batch, targets_batch
)

C0_train, C_O_train, t_s_train, T_train, targets_train = train_data
C0_val,   C_O_val,   t_s_val,   T_val,   targets_val   = val_data

print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C0_train = C0_train.to(device)
C_O_train = C_O_train.to(device)
t_s_train = t_s_train.to(device)
T_train = T_train.to(device)
targets_train = targets_train.to(device)

C0_val = C0_val.to(device)
C_O_val = C_O_val.to(device)
t_s_val = t_s_val.to(device)
T_val = T_val.to(device)
targets_val = targets_val.to(device)

model = model_class(hidden_size=hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
best_model_path = f'kconde_{args.model}_best.pth'


print("Start training...")
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model.integrate_batch(C0_train, t_s_train, C_O_train, T_train)
    loss_train = loss_fn(y_pred_train, targets_train)
    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_val = model.integrate_batch(C0_val, t_s_val, C_O_val, T_val)
            mse_val, rmse_pct, mae_pct = compute_metrics(y_pred_val, targets_val)
            print(f"Epoch {epoch+1:4d}/{args.epochs}, Train Loss: {loss_train.item():.6f} |"
                  f"Val Loss: {mse_val:.6f} | Val RMSE(%) : {rmse_pct:.2f} | Val MAE(%) : {mae_pct:.2f}")

torch.save(model.state_dict(), f'kconde_{args.model}_final.pth')
print(f"Training finished. Final model saved as kconde_{args.model}_final.pth")
print(f"Best model saved as {best_model_path} with val loss = {best_val_loss:.6f}")