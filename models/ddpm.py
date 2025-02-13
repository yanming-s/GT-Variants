import torch
import torch.nn as nn


def sym_tensor(x):
    x = x.permute(0,3,1,2) # [bs, d, n, n]
    triu = torch.triu(x,diagonal=1).transpose(3,2) # [bs, d, n, n]
    mask = (triu.abs()>0).float()                  # [bs, d, n, n]
    x =  x * (1 - mask ) + mask * triu             # [bs, d, n, n]
    x = x.permute(0,2,3,1) # [bs, n, n, d]
    return x               # [bs, n, n, d]


class DDPM(nn.Module):
    def __init__(self, num_t, beta_1, beta_T, UNet, num_atom_type, num_bond_type, device):
        super().__init__()
        self.num_t = num_t
        self.alpha_t = 1.0 - torch.linspace(beta_1, beta_T, num_t).to(device)
        self.alpha_bar_t = torch.cumprod( self.alpha_t, dim=0)
        self.UNet = UNet
        self.device = device
        self.num_atom_type = num_atom_type
        self.num_bond_type = num_bond_type
    def forward_process(self, x0, e0, sample_t, noise_x0, noise_e0): # add noise
        x0 = torch.nn.functional.one_hot(x0, self.num_atom_type) # one hot encoding
        e0 = torch.nn.functional.one_hot(e0, self.num_bond_type) # one hot encoding
        bs2 = len(sample_t)
        sqrt_alpha_bar_t = self.alpha_bar_t[sample_t].sqrt() # [bs]
        sqrt_one_minus_alpha_bar_t = ( 1.0 - self.alpha_bar_t[sample_t] ).sqrt() # [bs]
        x_t = sqrt_alpha_bar_t.view(bs2,1,1) * x0 + sqrt_one_minus_alpha_bar_t.view(bs2,1,1) * noise_x0 # [bs, n, n_atom]
        e_t = sqrt_alpha_bar_t.view(bs2,1,1,1) * e0 + sqrt_one_minus_alpha_bar_t.view(bs2,1,1,1) * noise_e0 # [bs, n, n, n_bond]
        return x_t, e_t
    def backward_process(self, x_t, e_t, sample_t): # denoise
        noise_pred_x_t, noise_pred_e_t = self.UNet(x_t, e_t, sample_t) # [bs, 28, 28]
        return noise_pred_x_t, noise_pred_e_t
    def generate_process_ddpm(self, num_mol, size_mol):
        t = self.num_t - 1
        bs2 = num_mol
        n = size_mol
        batch_t = (t * torch.ones(bs2)).long().to(self.device)
        batch_x_t = torch.randn(bs2, n, self.num_atom_type).to(self.device) # t=T => t=T-1 in python
        batch_e_t = torch.randn(bs2, n, n, self.num_bond_type).to(self.device) # t=T => t=T-1 in python
        batch_e_t = sym_tensor(batch_e_t) 
        set_t = list(range(t-1,0,-1)); set_t = set_t + [0]
        for t_minus_one in set_t:
            batch_t_minus_one = (t_minus_one * torch.ones(bs2)).long().to(self.device)
            batch_noise_pred_x_t, batch_noise_pred_e_t = self.backward_process(batch_x_t, batch_e_t, batch_t)
            sigma_t = ( (1.0-self.alpha_bar_t[t_minus_one])/ (1.0-self.alpha_bar_t[t])* (1.0-self.alpha_bar_t[t]/self.alpha_bar_t[t_minus_one]) ).sqrt()
            c1 = self.alpha_bar_t[t_minus_one].sqrt() / self.alpha_bar_t[t].sqrt()
            c2 = ( 1.0 - self.alpha_bar_t[t] + 1e-10 ).sqrt()
            c3 = ( 1.0 - self.alpha_bar_t[t_minus_one] - sigma_t.square() + 1e-10 ).sqrt()
            batch_x_t_minus_one = c1 * ( batch_x_t - c2 * batch_noise_pred_x_t ) + \
                c3 * batch_noise_pred_x_t + sigma_t* torch.randn(bs2, n, self.num_atom_type).to(self.device)
            noise_e_t = torch.randn(bs2, n, n, self.num_bond_type).to(self.device)
            noise_e_t = sym_tensor(noise_e_t)
            batch_e_t_minus_one = c1 * ( batch_e_t - c2 * batch_noise_pred_e_t ) + \
                c3 * batch_noise_pred_e_t + sigma_t* noise_e_t
            t = t_minus_one
            batch_x_t = batch_x_t_minus_one
            batch_e_t = batch_e_t_minus_one
            batch_t = batch_t_minus_one
        return batch_x_t, batch_e_t
