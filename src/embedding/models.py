from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, pairwise: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.pairwise = pairwise

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, key_padding_mask: Union[None, torch.Tensor] = None):
        B, N, E = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B,H,N,N

        if self.pairwise: # add pairwise bias only if enabled
            if pairwise_feats is None:
                raise ValueError("pairwise_feats must be provided when pairwise is True")
            bias_logits = self.bias_mlp(pairwise_feats)  # (B, N, N, H)
            bias_logits = bias_logits.permute(0, 3, 1, 2)  # (B, H, N, N)
            scores = scores + bias_logits
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # B,1,1,N
            scores = scores.masked_fill(mask == True, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # B,H,N,N
        out = torch.matmul(attn, V)  # B,H,N,head_dim

        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.out_proj(out)
        return out

class LinearAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, linear_dim: int, num_tokens: int, pairwise: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.pairwise = pairwise

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Linformer projection matrices
        self.f_proj = nn.Linear(num_tokens, linear_dim, bias=False)
        self.e_proj = nn.Linear(num_tokens, linear_dim, bias=False)

        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, key_padding_mask: Union[None, torch.Tensor] = None):
        B, N, E = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, self.head_dim)  # B,1,N,head_dim
            K = K.masked_fill(expanded_mask, 0.0)
            V = V.masked_fill(expanded_mask, 0.0)

        K_prime = self.e_proj(K.transpose(2, 3)).transpose(2, 3) # B,H,linear_dim,head_dim
        V_prime = self.f_proj(V.transpose(2, 3)).transpose(2, 3) # B,H,linear_dim,head_dim
        
        scores = torch.matmul(Q, K_prime.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,N,head_dim)x(B,H,head_dim,linear_dim) => B,H,N,linear_dim
        
        if self.pairwise: # add pairwise bias only if enabled
            if pairwise_feats is None:
                raise ValueError("pairwise_feats must be provided when pairwise is True")
            bias_logits = self.bias_mlp(pairwise_feats)  # (B, N, N, H)
            bias_logits = bias_logits.permute(0, 3, 1, 2)  # (B, H, N, N)
            bias_logits_prime = self.e_proj(bias_logits) # NEW. B,H,N,linear_dim

            scores = scores + bias_logits_prime

        attn = torch.softmax(scores, dim=-1)  # B,H,N,linear_dim
        out = torch.matmul(attn, V_prime)  # (B,H,N,linear_dim)x(B,H,linear_dim,head_dim) => B,H,N,head_dim

        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.out_proj(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            dim_feedforward: int = 2048, 
            dropout: float = 0.1, 
            linear_dim: Union[int, None] = None, 
            num_tokens: Union[int, None] = None,
            pairwise: bool = False
        ):
        super().__init__()
        if linear_dim is not None and num_tokens is None:
            raise ValueError("num_tokens must be provided if linear_dim is specified")
        self.self_attn = AttentionLayer(embed_dim, num_heads, pairwise) if linear_dim is None else LinearAttentionLayer(embed_dim, num_heads, linear_dim, num_tokens, pairwise)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
            self, 
            src: torch.Tensor, 
            pairwise_feats: Union[None, torch.Tensor] = None, 
            src_key_padding_mask: Union[None, torch.Tensor] = None
        ):
        src2 = self.self_attn(src, pairwise_feats, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    """
    Transformer encoder dimension parameters:
    - num_features: input feature dimension
    - embed_size: dimension of the token embeddings and the CLS token
    - latent_dim: dimension of the output latent representation (after bottleneck)
    - num_heads: number of attention heads per layer
    - num_layers: number of transformer layers
    - linear_dim: if specified, use linear attention with this projection dimension
    - num_tokens: if using linear attention, the maximum number of tokens (including CLS) for projection
    - pairwise: whether to use pairwise bias in attention layers (requires pairwise_feats input); resource intensive!
    """
    def __init__(
            self, 
            num_features: int, 
            embed_size: int, 
            latent_dim: int, 
            num_heads: int = 8, 
            num_layers: int = 4,
            linear_dim: Union[int, None] = None,
            num_tokens: Union[int, None] = None,
            pairwise: bool = False,
        ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_size, 
                    num_heads, 
                    linear_dim=linear_dim, 
                    num_tokens=num_tokens+1 if num_tokens is not None else None,
                    pairwise=pairwise
                ) for _ in range(num_layers)
            ]
        )
        self.norm_cls_embedding = nn.LayerNorm(embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.bottleneck = nn.Linear(embed_size, latent_dim)
        self.pairwise = pairwise # bool

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, mask: Union[None, torch.Tensor] = None):
        B, N, F = x.shape
        x = self.input_proj(x) # [B, N, E]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) 
        device = x.device

        if self.pairwise != (pairwise_feats is not None):
            raise ValueError(
                f"Pairwise mode is {self.pairwise}, but pairwise_feats was "
                f"{'provided' if pairwise_feats is not None else 'not provided'}"
            )
        
        pairwise_bias = None
        if self.pairwise:
            N = x.size(1)
            B = x.size(0)
            pairwise_bias = torch.zeros(B, N, N, 1, device=device)
            pairwise_bias[:, 1:, 1:, 0] = pairwise_feats[..., 0]
            
        for layer in self.layers:
            x = layer(x, pairwise_bias, src_key_padding_mask=mask)

        cls_embedding = x[:, 0, :] # CLS token embedding
        latent = self.bottleneck(self.norm_cls_embedding(cls_embedding))
        return latent
    
class Projector(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, z):
        z = self.net(z)
        z = F.normalize(z, dim=-1)
        return z
    
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, token_mlp_dim, channel_mlp_dim, dropout=0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp_1 = nn.Linear(num_tokens, token_mlp_dim)
        self.token_act = nn.ReLU()
        self.token_mlp_2 = nn.Linear(token_mlp_dim, num_tokens)
        self.token_dropout = nn.Dropout(dropout)

        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channel_mlp_dim),
            nn.ReLU(),
            nn.Linear(channel_mlp_dim, hidden_dim)
        )
        self.channel_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Token-mixing
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mlp_1(y)
        y = self.token_act(y)
        y = self.token_mlp_2(y)
        y = self.token_dropout(y)
        y = y.transpose(1, 2)
        x = x + y

        # Channel-mixing
        y = self.channel_mlp(x)
        y = self.channel_dropout(y)
        x = x + y
        return x

class MLPMixer(nn.Module):
    def __init__(self, num_particles, num_features, num_classes, num_blocks, token_mlp_dim, channel_mlp_dim, dropout=0.1):
        super().__init__()
        hidden_dim = num_features
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(num_particles, hidden_dim, token_mlp_dim, channel_mlp_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.avgpool = nn.AvgPool1d(num_particles)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        for block in self.mixer_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class EvalMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
