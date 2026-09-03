import math
import torch
import torch.nn as nn

class Degradation(nn.Module):
    """Simulates detector dead zones by randomly dropping particles whose
    (eta, phi) falls in a set grid/histogram of chosen 'dead' bins.

    It just zeroes the entire feature row of any particle it drops.

    This layer has to come before the PreProcessor layer.
    """

    def __init__(
        self,
        # The severity should be on a scale of 0 to 1 (no degradation to maximal degradation)
        severity: float = 0.0
    ):
        """The following initialization builds one histogram of detector defaults 
        for a given severity which is applied to all events for evaluation.
    
        num_miss is the number of bins that have a detector default.
        
        p_miss is the probability of a particle being dropped if it falls 
        inside a bin that has a detector default.
    
        The function creates histograms with 0s everywhere except in faulty bins. 
        In the bad bins, the histograms have value of p_miss sampled from
        a normal distribution with mean p_miss_mean and std p_miss_std.
        """
        super().__init__()
        self.n_eta_bins = 10
        self.n_phi_bins = 10
        self.n_bins = self.n_eta_bins * self.n_phi_bins
        # num_miss is the number of bins that have a detector default
        num_miss = severity * (self.n_eta_bins * self.n_phi_bins)
        self.num_miss = int(min(max(num_miss, 0), self.n_bins))
        self.p_miss_mean = 0.75
        self.p_miss_std = 0.1
        self.grid_seed = 42

        self.register_buffer("grid", torch.zeros(self.n_bins))

        if self.num_miss > 0:
            gen = torch.Generator().manual_seed(self.grid_seed)
            bad_idx = torch.randperm(self.n_bins, generator=gen)[:self.num_miss]
            values = torch.empty(self.num_miss) \
                          .normal_(self.p_miss_mean, self.p_miss_std, generator=gen) \
                          .clamp_(0.0, 1.0)
            self.grid[bad_idx] = values

        eta_range = (-5.0, 5.0)
        phi_range = (-math.pi, math.pi)
        eta_edges = torch.linspace(eta_range[0], eta_range[1], self.n_eta_bins + 1)
        phi_edges = torch.linspace(phi_range[0], phi_range[1], self.n_phi_bins + 1)
        self.register_buffer("eta_edges", eta_edges)
        self.register_buffer("phi_edges", phi_edges)

    def _dead_zone_drop_mask(self, eta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """For every particle, look up its (eta, phi) bin's miss-probability
        (own event's grid) and Bernoulli-sample whether it gets dropped.
        Returns bool [B, N].
        """
        B = eta.shape[0]
        eta_bin = torch.bucketize(eta, self.eta_edges, right=True) - 1
        eta_bin = eta_bin.clamp(0, self.n_eta_bins - 1)
        phi_bin = torch.bucketize(phi, self.phi_edges, right=True) - 1
        phi_bin = phi_bin.clamp(0, self.n_phi_bins - 1)

        flat_idx = eta_bin * self.n_phi_bins + phi_bin  # [B, N]

        prob_grid = self.grid.unsqueeze(0).expand(B, -1)
        miss_prob = torch.gather(prob_grid, 1, flat_idx)  # [B, N]
        return torch.bernoulli(miss_prob).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape [B, N, F] with features [pt, eta, phi, dxy, dxysig, is_pf, pdgId, ...]
        Returns x with the same shape but rows dropped by the dead-zone mask
        are zeroed out.
        """
        pt_raw = x[..., 0]
        eta_raw = x[..., 1]
        phi_raw = x[..., 2]

        valid = pt_raw > 0  # [B, N] real particles vs. existing padding

        if valid.any():
            drop = self._dead_zone_drop_mask(eta_raw, phi_raw)
            drop = drop & valid

            if drop.any():
                x = x.clone()
                x[drop] = 0.0

        return x