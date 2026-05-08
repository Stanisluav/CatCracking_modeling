import torch
import torch.nn as nn
from torchdiffeq import odeint

class KCONDE_FCC(nn.Module):
    def __init__(self, n_comp=5, n_reactions=6, hidden_size=16, T_ref=530.0):
        super().__init__()
        self.n_reactions = n_reactions
        self.T_ref = T_ref  # опорная температура, K
        self.R = 8.314

        # Обучаемые параметры: предэкспоненты и энергии активации для каждой реакции
        self.A = nn.Parameter(torch.zeros(n_reactions))        # ln(k_ref)
        self.Ea = nn.Parameter(torch.zeros(n_reactions))               # энергия активации, Дж/моль

        # Нейросеть для поправочных множителей (по одному на реакцию)
        # Вход: концентрации (5) + C_O + T (нормированные)
        self.ann = nn.Sequential(
            nn.Linear(n_comp + 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_reactions),
            nn.Softplus()
        )

        self.gamma = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def get_rate_constants(self, T):
        T_K = T + 273.15
        T_ref_K = self.T_ref + 273.15
        exponent = -self.Ea / self.R * (1.0 / T_K - 1.0 / T_ref_K)  # (batch, 6)
        k = torch.exp(self.A.unsqueeze(0) + exponent)       # (batch, 6)
        return k

    def predict_correction(self, y, C_O, T):
        """
        Возвращает поправочные множители φ1..φ6 (batch, 6)
        """
        batch = y.shape[0]
        device = y.device
        if C_O.dim() == 0:
            C_O = C_O.unsqueeze(0).expand(batch, 1)
        if T.dim() == 0:
            T = T.unsqueeze(0).expand(batch, 1)
        inputs = torch.cat([y, C_O, T], dim=1)
        return self.ann(inputs)

    def integrate_batch(self, C0_batch, t_final_batch, C_O_batch, T_batch):
        batch = C0_batch.shape[0]
        device = C0_batch.device

        self._C_O = C_O_batch.reshape(batch, 1)
        self._T = T_batch.reshape(batch, 1)

        T_real = self._T * 100.0 + 500.0   # (batch, 1)

        k = self.get_rate_constants(T_real)  # (batch, 6)

        def ode_func(t, y):
            # y: (batch, 5)
            phi = self.predict_correction(y, self._C_O, self._T)  # (batch, 6)
            # Эффективные скорости = k * φ (поэлементно)
            effective_rates = k * phi      # (batch, 6)
            v1 = effective_rates[:, 0]
            v2 = effective_rates[:, 1]
            v3 = effective_rates[:, 2]
            v4 = effective_rates[:, 3]
            v5 = effective_rates[:, 4]
            v6 = effective_rates[:, 5]

            C_VGO, C_LCO, C_gas, C_light, C_coke = y.unbind(dim=1)

            exp_gamma_t = torch.exp(-self.gamma * C_coke)

            dVGO = -(v1 + v2 + v3 + v4) * (C_VGO ** 2) * exp_gamma_t
            dLCO = (v1 * (C_VGO ** 2) - v5 * C_LCO) * exp_gamma_t
            dGas = (v2 * (C_VGO ** 2) + v5 * C_LCO - v6 * C_gas) * exp_gamma_t
            dLight = (v3 * (C_VGO ** 2) + v6 * C_gas) * exp_gamma_t
            dCoke = (v4 * (C_VGO ** 2)) * exp_gamma_t
            return torch.stack([dVGO, dLCO, dGas, dLight, dCoke], dim=1)

        max_t = t_final_batch.max().item()
        t_eval = torch.linspace(0, max_t, 200, device=device)
        sol = odeint(ode_func, C0_batch, t_eval, method='dopri5')
        # Выбираем значения, ближайшие к t_final_batch
        t_eval_tensor = t_eval.unsqueeze(1)
        indices = torch.argmin(torch.abs(t_eval_tensor - t_final_batch), dim=0)
        y_pred = sol[indices, torch.arange(batch), :]
        y_pred = torch.clamp(y_pred, min=0.0)
        return y_pred