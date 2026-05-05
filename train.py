import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataset import load_experimental_data, normalize_concentrations
from kconde_model import KCONDE_FCC

# Загрузка данных
experiments = load_experimental_data()
C_O_list = []
t_s_list = []
T_list = []
targets_norm = []

for exp in experiments:
    C_O_list.append(exp['C_O'])
    t_s_list.append(exp['t_s'])
    T_list.append(exp['T_C'])
    targets_norm.append(torch.tensor(normalize_concentrations(exp['target']), dtype=torch.float32))

C_O_batch = torch.tensor(C_O_list, dtype=torch.float32)
t_s_batch = torch.tensor(t_s_list, dtype=torch.float32)
T_batch = torch.tensor(T_list, dtype=torch.float32)
targets_batch = torch.stack(targets_norm)  # (N, 5)

C_O_batch = C_O_batch / 10.0   # т.к. C_O до 7
T_batch = (T_batch - 500.0) / 100.0   # центрирование

# Начальные условия: VGO=1, остальное 0
C0_batch = torch.zeros(len(experiments), 5)
C0_batch[:, 0] = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C0_batch = C0_batch.to(device)
C_O_batch = C_O_batch.to(device)
t_s_batch = t_s_batch.to(device)
T_batch = T_batch.to(device)
targets_batch = targets_batch.to(device)


model = KCONDE_FCC(hidden_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
epochs = 1500
for epoch in range(epochs):
    optimizer.zero_grad()
    # Интегрируем весь батч за один раз
    y_pred = model.integrate_batch(C0_batch, t_s_batch, C_O_batch, T_batch)
    loss = loss_fn(y_pred, targets_batch)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), 'kconde.pth')
print("Модель сохранена")