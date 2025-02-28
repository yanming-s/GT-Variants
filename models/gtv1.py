import torch
import torch.nn as nn


def sym_tensor(x):
    x = x.permute(0,3,1,2) # [bs, d, n, n]
    triu = torch.triu(x,diagonal=1).transpose(3,2) # [bs, d, n, n]
    mask = (triu.abs()>0).float()                  # [bs, d, n, n]
    x =  x * (1 - mask ) + mask * triu             # [bs, d, n, n]
    x = x.permute(0,2,3,1) # [bs, n, n, d]
    return x               # [bs, n, n, d]


class head_attention(nn.Module):
    def __init__(self, d, d_head, drop=0.0):
        super().__init__()
        self.Q = nn.Linear(d, d_head)
        self.K = nn.Linear(d, d_head)
        self.E = nn.Linear(d, d_head)
        self.V = nn.Linear(d, d_head)
        self.sqrt_d = torch.sqrt(torch.tensor(d_head))
        self.drop_att = nn.Dropout(drop)
        self.Ni = nn.Linear(d, d_head)
        self.Nj = nn.Linear(d, d_head)
    def forward(self, x, e):
        Q = self.Q(x) # [bs, n, d_head]
        K = self.K(x) # [bs, n, d_head]
        V = self.V(x) # [bs, n, d_head]
        Q = Q.unsqueeze(2) # [bs, n, 1, d_head]
        K = K.unsqueeze(1) # [bs, 1, n, d_head]
        E = self.E(e) # [bs, n, n, d_head]
        Ni = self.Ni(x).unsqueeze(2) # [bs, n, 1, d_head]
        Nj = self.Nj(x).unsqueeze(1) # [bs, 1, n, d_head]
        e = Ni + Nj + E              # [bs, n, n, d_head]
        Att = (Q * e * K).sum(dim=3) / self.sqrt_d # [bs, n, n]
        Att = torch.softmax(Att, dim=1)            # [bs, n, n]
        Att = self.drop_att(Att)                   # [bs, n, n]
        x = Att @ V                  # [bs, n, d_head]
        return x, e                  # [bs, n, d_head], [bs, n, n, d_head]


class MHA(nn.Module):
    def __init__(self, d, num_heads, drop=0.0):  
        super().__init__()
        d_head = d // num_heads
        self.heads = nn.ModuleList([head_attention(d, d_head, drop) for _ in range(num_heads)])
        self.WOx = nn.Linear(d, d)
        self.WOe = nn.Linear(d, d)
        self.drop_x = nn.Dropout(drop)
        self.drop_e = nn.Dropout(drop)
    def forward(self, x, e):
        x_MHA = []
        e_MHA = []    
        for head in self.heads:
            x_HA, e_HA = head(x,e)            # [bs, n, d_head], [bs, n, n, d_head]
            x_MHA.append(x_HA)
            e_MHA.append(e_HA)
        x = self.WOx(torch.cat(x_MHA, dim=2)) # [bs, n, d]
        x = self.drop_x(x)                    # [bs, n, d]
        e = self.WOe(torch.cat(e_MHA, dim=3)) # [bs, n, n, d]
        e = self.drop_e(e)                    # [bs, n, n, d]
        return x, e                           # [bs, n, d], [bs, n, n, d]


class BlockGT(nn.Module):
    def __init__(self, d, num_heads, drop=0.0):  
        super().__init__()
        self.LNx = nn.LayerNorm(d)
        self.LNe = nn.LayerNorm(d)
        self.LNx2 = nn.LayerNorm(d)
        self.LNe2 = nn.LayerNorm(d)
        self.MHA = MHA(d, num_heads, drop)
        self.MLPx = nn.Sequential(nn.Linear(d, 4*d), nn.LeakyReLU(), nn.Linear(4*d, d))
        self.MLPe = nn.Sequential(nn.Linear(d, 4*d), nn.LeakyReLU(), nn.Linear(4*d, d))
        self.drop_x_mlp = nn.Dropout(drop)
        self.drop_e_mlp = nn.Dropout(drop)
    def forward(self, x, e):
        x = self.LNx(x)                 # [bs, n, d]
        e = self.LNe(e)                 # [bs, n, n, d]
        x_MHA, e_MHA = self.MHA(x, e)   # [bs, n, d], [bs, n, n, d]
        x = x + x_MHA                   # [bs, n, d]
        x = x + self.MLPx(self.LNx2(x)) # [bs, n, d]
        x = self.drop_x_mlp(x)          # [bs, n, d]
        e = e + e_MHA                   # [bs, n, n, d]
        e = e + self.MLPe(self.LNe2(e)) # [bs, n, n, d]
        e = self.drop_e_mlp(e)          # [bs, n, n, d]
        return x, e                     # [bs, n, d], [bs, n, n, d]


class UNet_vanilla(nn.Module):
    def __init__(self, d, num_heads, num_layers, num_atom_type, num_bond_type, num_t, max_mol_sz, dPEt, device, drop=0.0):
        super().__init__()
        self.pe_x = nn.Embedding(max_mol_sz, d)
        self.atom_emb = nn.Linear(num_atom_type, d)
        self.bond_emb = nn.Linear(num_bond_type, d)
        self.pe_t_emb = nn.Sequential(nn.Embedding(num_t, dPEt), nn.ReLU(), nn.Linear(dPEt, d))
        self.gt_layers = nn.ModuleList([BlockGT(d, num_heads, drop) for _ in range(num_layers)])
        self.atom_dec = nn.Linear(d, num_atom_type)
        self.bond_dec = nn.Linear(d, num_bond_type)
        self.drop_x_emb = nn.Dropout(drop)
        self.drop_e_emb = nn.Dropout(drop)
        self.drop_p_emb = nn.Dropout(drop)
        self.device = device

    def forward(self, x_t, e_t, sample_t):
        bs2 = x_t.size(0); n = x_t.size(1)
        pe_x = torch.arange(0,n).to(self.device).repeat(bs2,1) # [bs, n]  
        pe_x = self.pe_x(pe_x)                            # [bs, n, d]  
        x_t = self.atom_emb(x_t)                          # [bs, n, d]
        x_t = x_t + pe_x                                  # [bs, n, d]
        e_t = self.bond_emb(e_t)                   # [bs, n, n, d]
        e_t = e_t + pe_x.unsqueeze(1) + pe_x.unsqueeze(2) # [bs, n, n, d]  
        e_t = sym_tensor(e_t)                      # [bs, n, n, d]
        p_t = self.pe_t_emb(sample_t)              # [bs, d]
        x_t = self.drop_x_emb(x_t)                 # [bs, n, d] 
        e_t = self.drop_e_emb(e_t)                 # [bs, n, n, d]
        p_t = self.drop_p_emb(p_t)                 # [bs, d]
        for gt_layer in self.gt_layers:
            x_t = x_t + p_t.unsqueeze(1)               # [bs, n, d]
            e_t = e_t + p_t.unsqueeze(1).unsqueeze(2)  # [bs, n, n, d] 
            x_t, e_t = gt_layer(x_t, e_t)          # [bs, n, d], [bs, n, n, d] 
            e_t = sym_tensor(e_t)                  # [bs, n, n, d] 
        x_t_minus_one = self.atom_dec(x_t)         # [bs, n, num_atom_type]
        e_t_minus_one = self.bond_dec(e_t)         # [bs, n, n, num_bond_type]
        return x_t_minus_one, e_t_minus_one


class DiGressNet_GTv1(nn.Module):
    def __init__(self, atom_dict: dict, bond_dict: dict, n_layers: int,
                 input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in, act_fn_out, device):
        super().__init__()
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.n_layers = n_layers
        self.out_dim_X = output_dims["x"]
        self.out_dim_E = output_dims["e"]
        self.device = device
        hidden_dims["dx"] = hidden_dims["de"]
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["x"], hidden_mlp_dims["x"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["x"], hidden_dims["dx"]), act_fn_in
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["e"], hidden_mlp_dims["e"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["e"], hidden_dims["de"]), act_fn_in
        )
        self.tf_layers = nn.ModuleList(
            [
                BlockGT(
                    d=hidden_dims['dx'],
                    num_heads=hidden_dims['n_head'],
                    drop=0.0
                )
                for _ in range(n_layers)
            ]
        )
        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['x']), act_fn_out,
            nn.Linear(hidden_mlp_dims['x'], output_dims['x'])
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims['de'], hidden_mlp_dims['e']), act_fn_out,
            nn.Linear(hidden_mlp_dims['e'], output_dims['e'])
        )
    def forward(self, noise_x: torch.Tensor, noise_e: torch.Tensor, extra_x: torch.Tensor, _, extra_y: torch.Tensor):
        X = noise_x.float().to(self.device)
        E = noise_e.float().to(self.device)
        X = torch.cat((X, extra_x.to(self.device)), dim=2).float()
        # E = torch.cat((E, extra_e), dim=3).float() # extra_e is all zeros
        X_res = X[..., :self.out_dim_X]
        E_res = E[..., :self.out_dim_E]
        new_E = self.mlp_in_E(E)
        E = (new_E + new_E.transpose(1, 2)) / 2
        X = self.mlp_in_X(X)
        for layer in self.tf_layers:
            X, E = layer(X, E)
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        X = (X + X_res)
        E = (E + E_res)
        E = 1/2 * (E + torch.transpose(E, 1, 2))
        return X, E
