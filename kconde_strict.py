import torch
import torch.nn as nn
from torchdiffeq import odeint

class KCONDE_Strict(nn.Module):
    def __init__(self, n_comp=5, n_reactions=6, hidden_size=12):
        super().__init__()
        # Аррениус
        self.log_k0_ref = nn.Parameter(torch.zeros(n_reactions))
        self.Ea = nn.Parameter(torch.randn(n_reactions) * 5.0)
        self.register_buffer('T_ref', torch.tensor(800.0))
        
        # Деактивация (пока отключен)
        # self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.net = nn.Sequential(
            nn.Linear(n_comp + 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_reactions),
            nn.Softplus()  # множитель >0
        )
        
    def get_rate_constants(self, T_abs):
        R = 0.008314
        inv_T = 1.0 / T_abs
        inv_T_ref = 1.0 / self.T_ref
        exponent = -self.Ea / R * (inv_T - inv_T_ref)
        k = torch.exp(self.log_k0_ref) * torch.exp(exponent)
        return k  # (batch, 6)
    
    def forward(self, t, state, C_O, T_abs):
        state = torch.clamp(state, 0.0, 1.0)
        C_VGO, C_LCO, C_gas, C_light, C_coke = state.unbind(dim=1)
        
        k = self.get_rate_constants(T_abs)  # (batch, 6)
        # Пока без деактивации: phi = 1
        phi = 1.0

        r_base_1 = k[:, 0] * C_VGO * phi
        r_base_2 = k[:, 1] * C_VGO * phi
        r_base_3 = k[:, 2] * C_VGO * phi
        r_base_4 = k[:, 3] * C_VGO * phi
        r_base_5 = k[:, 4] * C_LCO * phi
        r_base_6 = k[:, 5] * C_gas * phi
        
        T_norm = T_abs / 800.0
        inputs = torch.cat([state, C_O, T_norm], dim=1)  # (batch, 7)
        f = self.net(inputs)      # поправочные множители для каждой реакции
        
        r1 = r_base_1 * f[:, 0]
        r2 = r_base_2 * f[:, 1]
        r3 = r_base_3 * f[:, 2]
        r4 = r_base_4 * f[:, 3]
        r5 = r_base_5 * f[:, 4]
        r6 = r_base_6 * f[:, 5]
        
        dVGO   = -(r1 + r2 + r3 + r4)
        dLCO   =  r1 - r5
        dGas   =  r2 + r5 - r6
        dLight =  r3 + r6
        dCoke  =  r4
        
        return torch.stack([dVGO, dLCO, dGas, dLight, dCoke], dim=1)
    
    def integrate_batch(self, C0_batch, t_final_batch, C_O_batch, T_abs_batch):
        batch = C0_batch.shape[0]
        device = C0_batch.device
        C_O_batch = C_O_batch.view(batch,1)
        T_abs_batch = T_abs_batch.view(batch,1)
        
        max_t = t_final_batch.max().item()
        if max_t <= 1e-6:
            return C0_batch
        t_eval = torch.linspace(0, max_t, 200, device=device)
        
        def ode_func(t, y):
            y = torch.clamp(y, 0.0, 1.0)
            return self.forward(t, y, C_O_batch, T_abs_batch)
        
        sol = odeint(ode_func, C0_batch, t_eval, method='dopri5', rtol=1e-3, atol=1e-4)
        
        # Интерполяция
        indices = torch.argmin(torch.abs(t_eval.unsqueeze(1) - t_final_batch), dim=0)
        y_pred = sol[indices, torch.arange(batch), :]
        return torch.clamp(y_pred, 0.0, 1.0)