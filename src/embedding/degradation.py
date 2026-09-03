import torch
import torch.nn as nn

class Degradation(nn.Module):
    """Simulate your detector degradation"""

    def __init__(
        self,
        # Your degradation should be a function of severity
        # Severity of 0 should represent no degradation
        # Severity can then scale however you would like 
        severity: float = 0.0,
    ):
        super().__init__()
        self.severity = severity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape [batch size, number particles, number of features] .
        The features in order are [pt, eta, phi, dxy, dxysig, is_pf, pdgId, ...].
        Returns x with the degradation applied
        """

        # Add your degradation code here!

        # You can access the features like this:
        pt = x[..., 0]
        eta = x[..., 1]
        phi = x[..., 2]

        return x