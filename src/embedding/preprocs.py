import torch
import torch.nn as nn
from typing import Union
from embedding.utils.data_utils import EPS

class PFPreProcessor(nn.Module):
    def __init__(self, norm_constants: dict = {}):
        super().__init__()
        self.norm_constants = norm_constants
        PDGIDs = [
            211, # h, charged hadrons
            11,  # e
            13,  # mu
            22,  # gamma
            130, # h0
            1,   # h_HF, HF tower identified as a hadron
            2    # egamma_HF, HF tower identified as an EM particle
        ]
        self.register_buffer("avail_pdgIds", torch.tensor(PDGIDs, dtype=torch.long))
        self.num_features_cont = 5 # pt, eta, phi, dxy, dxysig
        self.num_features_disc = 2 + len(self.avail_pdgIds) # is_pf, charge(from pdgId sign), pdgId(one-hot)
        self.num_features = self.num_features_cont + self.num_features_disc
        self.batch_norm = nn.BatchNorm1d(self.num_features_cont)

    def pdgId_to_onehot(self, pdgId_tensor: torch.Tensor) -> torch.Tensor:
        pdgId_tensor = pdgId_tensor.long()  # [B, N]
        one_hot = (pdgId_tensor.unsqueeze(-1).abs() == self.avail_pdgIds).float()  # [B, N, num_pdgIds]
        return one_hot
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, 8] = [pt, eta, phi, dxy, dxysig, is_pf, pdgId]
        Returns: [B, N, 8 + num_pdgIds] with normalized/scaled values and one-hot pdgId.
        - pt:        log(pt / sum_pt) per event
        - eta:       kept as is
        - phi:       kept as is
        - dxy:       tanh(dxy)
        - dxysig:    kept as is
        - charge:    derived from pdgId: +1/-1 for e, mu, pi; 0 else
        - is_pf:     kept as is (0/1)
        - pdgId:     one-hot encoded absolute pdgId from available list
        Padding (pt==0) rows are zeroed.
        Continuous features are batch-normalized.
        """

        pt_raw = x[..., 0]
        eta_raw = x[..., 1]
        phi_raw = x[..., 2]
        dxy_raw = x[..., 3]
        dxysig_raw = x[..., 4]
        is_pf_raw = x[..., 5]
        pdgId_raw = x[..., 6]

        valid = pt_raw > 0 # [B, N]
        
        if valid.any():
            pt = torch.where(valid, pt_raw, torch.zeros_like(pt_raw))
            pt = pt / (pt.sum(dim=-1, keepdim=True) + EPS)
            pt[valid] = torch.log(pt[valid]) 
            
            dxy = torch.where(valid, dxy_raw, torch.zeros_like(dxy_raw))
            dxy[valid] = torch.tanh(dxy_raw[valid])

            pos = (pdgId_raw == 11) | (pdgId_raw == 13) | (pdgId_raw == 211)
            neg = (pdgId_raw == -11) | (pdgId_raw == -13) | (pdgId_raw == -211)
            charge = torch.zeros_like(valid, dtype=torch.float)
            charge[valid & pos] = 1.0
            charge[valid & neg] = -1.0
            
            pdgId_onehot = self.pdgId_to_onehot(pdgId_raw)
            
            x_proc = torch.cat([
                pt.unsqueeze(-1),
                eta_raw.unsqueeze(-1),
                phi_raw.unsqueeze(-1),
                dxy.unsqueeze(-1),
                dxysig_raw.unsqueeze(-1),
                # Next ones NOT are not continuous, so NOT fed to batch norm layer.
                charge.unsqueeze(-1), # Computed
                is_pf_raw.unsqueeze(-1),
                pdgId_onehot
            ], dim=-1)

            # Masked batch norm on continuous ftrs
            x_cont = x_proc[..., :self.num_features_cont]  # [B, N, num_cont]
            B, N, C = x_cont.shape
            x_cont_flat = x_cont.reshape(B * N, C)
            valid_flat = valid.reshape(B * N)
            x_cont_flat[valid_flat] = self.batch_norm(x_cont_flat[valid_flat])
            x_proc[..., :self.num_features_cont] = x_cont_flat.reshape(B, N, C)
        else:
            x_proc = torch.zeros(*x.shape[:-1], self.num_features, device=x.device)
        
        return x_proc

class PUPPIPreProcessor(nn.Module):
    def __init__(self, norm_constants: dict):
        super().__init__()
        self.norm_constants = norm_constants
        self.num_features = 7

    def forward(self, x):
        """
        x: [P, 7] = [pt, eta, phi, dxy, btag, has_dxy, has_btag]
        Returns: [P, 7] with normalized/scaled values and flags kept as channels.
        - pt:   log1p + min–max (train-split stats)
        - eta:  min–max
        - phi:  wrapped to (-pi, pi] then scaled to [0,1]
        - dxy:  min–max where defined (has_dxy), else 0
        - btag: clamped to [0,1] where defined (has_btag), else 0
        - flags: kept as float channels
        Padding (pt==0) rows are zeroed.
        """

        pt_raw   = x[:, 0]
        eta_raw  = x[:, 1]
        phi_raw  = x[:, 2]
        dxy_raw  = x[:, 3]
        btag_raw = x[:, 4]
        has_dxy  = (x[:, 5] > 0.5)
        has_btag = (x[:, 6] > 0.5)

        valid = pt_raw > 0

        # pt: 
        pt = torch.zeros_like(pt_raw)
        if valid.any():
            pt_log = torch.log1p(pt_raw[valid])
            pt[valid] = (pt_log - self.norm_constants["pt_min"]) / (self.norm_constants["pt_max"] - self.norm_constants["pt_min"] + EPS)

        # eta: min–max on valid
        eta = torch.zeros_like(eta_raw)
        if valid.any():
            eta[valid] = (eta_raw[valid] - self.norm_constants["eta_min"]) / (self.norm_constants["eta_max"] - self.norm_constants["eta_min"] + EPS)

        # phi:
        phi = torch.zeros_like(phi_raw)
        if valid.any():
            phi[valid] = (phi_raw[valid] + torch.pi) / (2 * torch.pi)

        # dxy: min–max only where defined (valid & has_dxy)
        dxy = torch.zeros_like(dxy_raw)
        dv = valid & has_dxy
        if dv.any():
            dxy[dv] = (dxy_raw[dv] - self.norm_constants["dxy_min"]) / (self.norm_constants["dxy_max"] - self.norm_constants["dxy_min"] + EPS)

        # btag: (valid & has_btag)
        btag = torch.zeros_like(btag_raw)
        jb = valid & has_btag
        if jb.any():
            btag[jb] = btag_raw[jb]

        has_dxy_f  = has_dxy.float()
        has_btag_f = has_btag.float()
        has_dxy_f[~valid] = 0.0
        has_btag_f[~valid] = 0.0

        normed = torch.stack([pt, eta, phi, dxy, btag, has_dxy_f, has_btag_f], dim=-1)
        return normed