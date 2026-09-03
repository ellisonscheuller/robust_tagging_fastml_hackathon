import torch
from typing import Union

EPS = 1e-4

def ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} has NaNs/Infs")

def clean_data(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(data).all():
        raise ValueError("Input data contains NaNs/Infs that could not be cleaned.")
    feature_block = data[..., :-1]
    label_block = data[:, 0, -1].long()

    if not torch.isfinite(feature_block).all():
        raise ValueError("feature_block contains NaNs/Infs after cleaning.")

    return feature_block, label_block

def load_data(path: str, map_location: torch.device, max_events=-1) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location=map_location)
    data = data[:max_events] if max_events > 0 else data
    return clean_data(data)

def compute_class_weights(label_block: torch.Tensor, setting: Union[None, str, list] = None) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequencies"""
    if setting is None:
        return torch.ones(int(label_block.max().item()) + 1, dtype=torch.float32)
    elif isinstance(setting, list):
        if len(setting) != int(label_block.max().item()) + 1:
            raise ValueError("Length of class_weights does not match number of classes")
        return torch.tensor(setting, dtype=torch.float32)
    elif isinstance(setting, str):
        if setting.lower() != "inv_freq":
            raise ValueError(f"Unknown class_weights setting: {setting}")
    else:
        raise ValueError("class_weights_setting must be None, a list, or 'inv_freq'")
    
    class_counts = torch.bincount(label_block)
    total_samples = label_block.size(0)
    num_classes = class_counts.size(0)
    class_weights = total_samples / ((class_counts.float() + EPS) * num_classes)
    class_weights = class_weights / class_weights.sum()
    return class_weights

def softkill(
        event_tensor: torch.Tensor, # Shape: [B, N, F]
        cell_size: float = 0.4, 
    ) -> torch.Tensor:
    """Apply soft-kill particles in the event tensor."""
    # TODO: Is this a good range for eta?
    # eta_max = 3.0
    # eta_min = -3.0
    eta_max = 5.0
    eta_min = -5.0
    phi_max = torch.pi
    phi_min = -torch.pi

    # Div eta-phi into grid of square cells of given cell_size
    eta_bins = torch.arange(eta_min, eta_max + cell_size, cell_size, device=event_tensor.device)
    phi_bins = torch.arange(phi_min, phi_max + cell_size, cell_size, device=event_tensor.device)
    
    # bin
    grid_indices_eta = torch.bucketize(event_tensor[...,1], eta_bins) - 1 # [B, N]
    grid_indices_phi = torch.bucketize(event_tensor[...,2], phi_bins) - 1 # [B, N]

    grid_shape = (len(eta_bins)-1, len(phi_bins)-1) # (number of eta cells, number of phi cells)
    max_pt_grid = torch.zeros((event_tensor.size(0), *grid_shape), device=event_tensor.device) # [B, num_eta_cells, num_phi_cells]
    max_pt_grid_flat = max_pt_grid.view(event_tensor.size(0), -1) # [B, num_eta_cells * num_phi_cells]
    linear_indices = grid_indices_eta * (len(phi_bins)-1) + grid_indices_phi # [B, N]

    pt = event_tensor[...,0] # [B, N]
    is_particle_masks = pt > 0 # [B, N]

    # TODO: Find way to vectorize this loop
    for i in range(event_tensor.size(0)):
        max_pt_grid_flat[i].scatter_reduce_(
            0,
            linear_indices[i][is_particle_masks[i]],
            pt[i][is_particle_masks[i]],
            reduce='amax',
            include_self=True
        )

    # get cuts and apply sk
    pt_cuts = (torch.median(max_pt_grid_flat, dim=1).values).unsqueeze(-1) # [B,1]
    pt_mask = pt >= pt_cuts # [B, N]

    event_tensor = event_tensor * pt_mask.unsqueeze(-1).float() # Zero out killed particles, [B, N, F]
    return event_tensor

def compute_delta_r(x):
    B, N, F = x.shape
    eta = x[:, :, 1] 
    phi = x[:, :, 2] 

    delta_eta = eta.unsqueeze(2) - eta.unsqueeze(1)  
    delta_phi = phi.unsqueeze(2) - phi.unsqueeze(1)  

    delta_phi = (delta_phi + torch.pi) % (2 * torch.pi) - torch.pi

    delta_r = torch.sqrt(delta_eta ** 2 + delta_phi ** 2) 
    delta_r = delta_r.unsqueeze(-1)  

    return delta_r

def delta_r_from_normalized(x, norm_constants):
    """
    Compute ΔR using physical eta/phi reconstructed from normalized x.
    x: [B, N, F], where x[...,1]=eta_norm in [0,1], x[...,2]=phi_norm in [0,1]
    """
    x_phys = x.clone()

    eta_min = norm_constants["eta_min"].to(x.device)
    eta_max = norm_constants["eta_max"].to(x.device)

    # denormalize
    x_phys[:, :, 1] = x[:, :, 1] * (eta_max - eta_min) + eta_min
    x_phys[:, :, 2] = x[:, :, 2] * (2 * torch.pi) - torch.pi
    # ensure in [-pi, pi]
    x_phys[:, :, 2] = (x_phys[:, :, 2] + torch.pi) % (2 * torch.pi) - torch.pi

    return compute_delta_r(x_phys)

def compute_normalization_constants(data):  
    pt = data[:, :, 0]
    eta = data[:, :, 1]
    phi = data[:, :, 2]
    dxy = data[:, :, 3]
    btag = data[:, :, 4]
    has_dxy  = data[:, :, 5] > 0.5
    has_btag = data[:, :, 6] > 0.5

    valid_pt = pt > 0

    if valid_pt.any():
        pt_log = torch.log1p(pt[valid_pt])
        pt_min, pt_max = pt_log.min(), pt_log.max()
    else:
        pt_min = torch.tensor(0.0)
        pt_max = torch.tensor(1.0)

    eta_min, eta_max = eta.min(), eta.max()
    phi_min, phi_max = -torch.pi, torch.pi  
    if has_dxy.any():
        dxy_min, dxy_max = dxy[has_dxy].min(), dxy[has_dxy].max()
    else:
        # fallback to zeros to avoid NaNs; will be unused anyway
        dxy_min, dxy_max = torch.tensor(0.0), torch.tensor(1.0) #check this def if its correct

    # if has_btag.any():
    #     btag_min, btag_max = btag[has_btag].min(), btag[has_btag].max()
    # else:
    #     btag_min, btag_max = torch.tensor(0.0), torch.tensor(1.0) #check this def if its correct

    return {
        'pt_min': pt_min,
        'pt_max': pt_max,
        'eta_min': eta_min,
        'eta_max': eta_max,
        'phi_min': phi_min,
        'phi_max': phi_max, 
        'dxy_min': dxy_min, 
        'dxy_max': dxy_max
    }

