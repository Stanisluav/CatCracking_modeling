import torch
import torch.nn as nn
from torchdiffeq import odeint

class KCONDE_FCC(nn.Module):
    def __init__(self, n_comp=5, n_reactions=6, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_comp + 2, hidden_size),  # 5 конц + C_O + T
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_reactions),
            nn.Softplus()  # положительные скорости
        )
        self.beta = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def predict_rates(self, y, C_O, T):
        # y: (batch, 5), C_O и T: скаляры или тензоры (batch,1)
        batch = y.shape[0]
        device = y.device
        # Если C_O и T переданы как скаляры, расширяем
        if C_O.dim() == 0:
            C_O = C_O.unsqueeze(0).expand(batch, 1)
        if T.dim() == 0:
            T = T.unsqueeze(0).expand(batch, 1)
        inputs = torch.cat([y, C_O, T], dim=1)  # (batch, 7)
        return self.net(inputs)  # (batch, 6)

    def forward(self, t, state):
        raise NotImplementedError("Используйте метод integrate_batch")

    def integrate_batch(self, C0_batch, t_final_batch, C_O_batch, T_batch):
        # C0_batch: (batch, 5) начальные концентрации (нормированные)
        # t_final_batch: (batch,) конечное время для каждого образца
        # C_O_batch: (batch,) или (batch,1)
        # T_batch: (batch,) или (batch,1)
        batch = C0_batch.shape[0]
        device = C0_batch.device
        
        def ode_func(t, y):
            # y: (batch, 5)
            rates = self.predict_rates(y, self._C_O, self._T)  # (batch,6)
            k1,k2,k3,k4,k5,k6 = rates.unbind(dim=1)
            phi = torch.exp(-self.beta * t)
            C_VGO, C_LCO, C_gas, C_light, C_coke = y.unbind(dim=1)
            dVGO = -(k1+k2+k3+k4) * (C_VGO**2) * phi
            dLCO = (k1 * (C_VGO**2) - k5 * C_LCO) * phi
            dGas = (k2 * (C_VGO**2) + k5 * C_LCO - k6 * C_gas) * phi
            dLight = (k3 * (C_VGO**2) + k6 * C_gas) * phi
            dCoke = (k4 * (C_VGO**2)) * phi
            return torch.stack([dVGO, dLCO, dGas, dLight, dCoke], dim=1)
        
        max_t = t_final_batch.max().item()
        t_eval = torch.linspace(0, max_t, 100, device=device)  # 100 шагов
        self._C_O = C_O_batch.reshape(batch,1)
        self._T = T_batch.reshape(batch,1)

        sol = odeint(ode_func, C0_batch, t_eval, method='dopri5')  # (len(t_eval), batch, 5)
        # Нахождение ближайшего индекса к t_final
        t_eval_tensor = t_eval.unsqueeze(1)  # (len,1)
        indices = torch.argmin(torch.abs(t_eval_tensor - t_final_batch), dim=0)
        y_pred = sol[indices, torch.arange(batch), :]
        return y_pred