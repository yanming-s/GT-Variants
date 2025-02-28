# Libraries
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import networkx as nx
import sys; sys.path.insert(0, "lib/")
from lib.molecules import Dictionary, MoleculeDataset, MoleculeDGL, Molecule, compute_ncut
import os, datetime
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import numpy as np
import math


from models.gtv2_film import BlockGT

TEST = False
DEVICE= torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main(test=False, device="cpu"):
    def group_molecules_per_size(dataset):
        mydict = {}
        for mol in dataset:
            if len(mol) not in mydict:
                mydict[len(mol)] = []
            mydict[len(mol)].append(mol)
        return mydict

    def sym_tensor(x):
        x = x.permute(0, 3, 1, 2)
        triu = torch.triu(x, diagonal=1).transpose(3, 2)
        mask = (triu.abs()>0).float()
        x =  x * (1 - mask) + mask * triu
        x = x.permute(0, 2, 3, 1)
        return x


    class MoleculeSampler:
        def __init__(self, organized_dataset, bs, shuffle=True):
            self.bs = bs
            self.num_mol =  {sz: len(list_of_mol) for sz, list_of_mol in organized_dataset.items()}
            self.counter = {sz: 0   for sz in organized_dataset}
            if shuffle:
                self.order = {sz: np.random.permutation(num)  for sz , num in self.num_mol.items()}
            else:
                self.order = {sz: np.arange(num)  for sz , num in self.num_mol.items()}

        def compute_num_batches_remaining(self):
            return {sz:  math.ceil(((self.num_mol[sz] - self.counter[sz])/self.bs))  for sz in self.num_mol}

        def choose_molecule_size(self):
            num_batches = self.compute_num_batches_remaining()
            possible_sizes = np.array(list(num_batches.keys()))
            prob = np.array(list(num_batches.values()))
            prob = prob / prob.sum()
            sz   = np.random.choice(possible_sizes, p=prob)
            return sz

        def is_empty(self):
            num_batches= self.compute_num_batches_remaining()
            return sum(num_batches.values()) == 0

        def draw_batch_of_molecules(self, sz):
            if (self.num_mol[sz] - self.counter[sz]) / self.bs >= 1.0:
                bs = self.bs
            else:
                bs = self.num_mol[sz] - (self.num_mol[sz] // self.bs) * self.bs
            indices = self.order[sz][self.counter[sz]:self.counter[sz] + bs]
            self.counter[sz] += bs
            return indices


    class GT(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_emb = nn.Embedding(num_atom_type, d)
            self.bond_emb = nn.Embedding(num_bond_type, d)
            num_layers_encoder = 4
            self.BlockGT_encoder_layers = nn.ModuleList([BlockGT(d, num_heads) for _ in range(num_layers_encoder)])
            self.ln_x_final = nn.LayerNorm(d)
            self.linear_x_final = nn.Linear(d, 1, bias=True)
            self.drop_x_emb = nn.Dropout(drop)
            self.drop_e_emb = nn.Dropout(drop)
        def forward(self, x, e):
            x = self.atom_emb(x) # [bs, n, d]
            e = self.bond_emb(e) # [bs, n, n, d]
            e = sym_tensor(e) # [bs, n, n, d]
            x = self.drop_x_emb(x)
            e = self.drop_e_emb(e)
            for gt_layer in self.BlockGT_encoder_layers:
                x, e = gt_layer(x, e)  # [bs, n, d], [bs, n, n, d]
                e = sym_tensor(e)
            mol_token = x.mean(1) # [bs, d]
            x = self.ln_x_final(mol_token)
            x = self.linear_x_final(x)
            return x


    data_folder_pytorch = "dataset/ZINC/"

    with open(data_folder_pytorch+"atom_dict.pkl","rb") as f:
        atom_dict=pickle.load(f)
    with open(data_folder_pytorch+"bond_dict.pkl","rb") as f:
        bond_dict=pickle.load(f)
    with open(data_folder_pytorch+"train.pkl","rb") as f:
        train=pickle.load(f)
    with open(data_folder_pytorch+"test.pkl","rb") as f:
        test=pickle.load(f)

    num_atom_type = len(atom_dict.idx2word)
    num_bond_type = len(bond_dict.idx2word)

    test_group  = group_molecules_per_size(test)
    train_group = group_molecules_per_size(train)

    num_heads = 8; d = 16 * num_heads; num_layers = 4; drop = 0.0; bs = 50

    num_mol_size = 20
    num_warmup = 2 * max(num_mol_size, len(train) // bs)

    if test:
        net = GT()
        net = net.to(device)
        def display_num_param(net):
            nb_param = 0
            for param in net.parameters():
                nb_param += param.numel()
            print('Number of parameters: {} ({:.2f} million)'.format(nb_param, nb_param/1e6))
            return nb_param/1e6
        _ = display_num_param(net)

        # Test the forward pass, backward pass and gradient update with a single batch
        init_lr = 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=init_lr)

        bs = 50
        sampler = MoleculeSampler(train_group, bs)
        print('sampler.num_mol :',sampler.num_mol)
        num_batches_remaining = sampler.compute_num_batches_remaining()
        print('num_batches_remaining :',num_batches_remaining)
        sz = sampler.choose_molecule_size()
        print('sz :',sz)
        indices = sampler.draw_batch_of_molecules(sz)
        print('indices :',len(indices),indices)
        batch_x0 = minibatch_node = torch.stack( [ train_group[sz][i].atom_type for i in indices] ).long().to(device) # [bs, n]
        print('minibatch_node :',minibatch_node.size())
        batch_e0 = minibatch_edge = torch.stack( [ train_group[sz][i].bond_type for i in indices] ).long().to(device) # [bs, n, n]
        print('minibatch_edge :',minibatch_edge.size())
        batch_target = torch.stack( [ train_group[sz][i].logP_SA_cycle_normalized for i in indices] ).float().to(device) # [bs, 1]
        print('batch_target :',batch_target.size())

        batch_x_pred = net(batch_x0, batch_e0) # [bs, 1]
        print('batch_x_pred',batch_x_pred.size())

        loss = nn.L1Loss()(batch_x_pred, batch_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    else:
    # Random seed
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Training loop
        net = GT()
        net = net.to(device)

        # Optimizer
        init_lr = 0.0001
        optimizer = torch.optim.Adam(net.parameters(), lr=init_lr)
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: min((t+1)/num_warmup, 1.0) ) # warmup scheduler
        scheduler_tracker = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=1) # tracker scheduler

        num_warmup_batch = 0

        # Number of mini-batches per epoch
        nb_epochs = 250
        lossMAE = nn.L1Loss()

        last_10_train_loss = []
        last_10_test_loss = []
        start=time.time()
        for epoch in range(nb_epochs):
            running_loss = 0.0
            num_batches = 0
            num_data = 0
            net.train()

            bs = 512
            sampler = MoleculeSampler(train_group, bs)
            while(not sampler.is_empty()):
                num_batches_remaining = sampler.compute_num_batches_remaining()
                sz = sampler.choose_molecule_size()
                indices = sampler.draw_batch_of_molecules(sz)
                bs2 = len(indices)
                batch_x0 = minibatch_node = torch.stack( [ train_group[sz][i].atom_type for i in indices] ).long().to(device) # [bs, n]
                batch_e0 = minibatch_edge = torch.stack( [ train_group[sz][i].bond_type for i in indices] ).long().to(device) # [bs, n, n]
                batch_target = torch.stack( [ train_group[sz][i].logP_SA_cycle_normalized for i in indices] ).float().to(device) # [bs, 1]
                batch_x_pred = net(batch_x0, batch_e0) # [bs, 1]
                loss = lossMAE(batch_x_pred, batch_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if num_warmup_batch < num_warmup:
                    scheduler_warmup.step() # warmup scheduler
                num_warmup_batch += 1

                # Compute stats
                running_loss += bs2 * loss.detach().item()
                num_batches += 1
                num_data += bs2

            # Test set
            bs = 512
            sampler = MoleculeSampler(test_group, bs)
            running_test_loss = 0
            num_test_data = 0
            with torch.no_grad():
                while(not sampler.is_empty()):
                    num_batches_remaining = sampler.compute_num_batches_remaining()
                    sz = sampler.choose_molecule_size()
                    indices = sampler.draw_batch_of_molecules(sz)
                    bs2 = len(indices)
                    batch_x0 = minibatch_node = torch.stack( [ test_group[sz][i].atom_type for i in indices] ).long().to(device) # [bs, n]
                    batch_e0 = minibatch_edge = torch.stack( [ test_group[sz][i].bond_type for i in indices] ).long().to(device) # [bs, n, n]
                    batch_target = torch.stack( [ test_group[sz][i].logP_SA_cycle_normalized for i in indices] ).float().to(device) # [bs, 1]
                    batch_x_pred = net(batch_x0, batch_e0) # [bs, 1]
                    running_test_loss += bs2 * lossMAE(batch_x_pred, batch_target).detach().item()
                    num_test_data += bs2

            # Average stats and display
            mean_train_loss = running_loss/num_data
            mean_test_loss = running_test_loss/num_test_data
            if nb_epochs - epoch <= 10:
                last_10_train_loss.append(mean_train_loss)
                last_10_test_loss.append(mean_test_loss)

            if num_warmup_batch >= num_warmup:
                scheduler_tracker.step(mean_train_loss) # tracker scheduler defined w.r.t. loss value
                num_warmup_batch += 1
            elapsed = (time.time()-start)/60

            if not epoch % 25:
                line = 'epoch= ' + str(epoch) + '\t time= ' + str(elapsed)[:6] + ' min' + '\t lr= ' + \
                '{:.7f}'.format(optimizer.param_groups[0]['lr']) + '\t train_loss= ' + str(mean_train_loss)[:6] + \
                '\t test_loss= ' + str(mean_test_loss)[:6]
                print(line)

        print()
        print("GTv2 FiLM")
        print(f"time: {elapsed:.4f} min")
        print(f"last 10 train loss - mean: {np.mean(last_10_train_loss):.4f} - std: {np.std(last_10_train_loss):.4f}")
        print(f"last 10 test loss - mean: {np.mean(last_10_test_loss):.4f} - std: {np.std(last_10_test_loss):.4f}")

    del net
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(TEST, DEVICE)
