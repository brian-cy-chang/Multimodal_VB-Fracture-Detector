"""
Adopted from https://github.com/jaketae/alibi
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y

def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

class ALiBiConfig:
    def __init__(self, num_layers=8, d_model=768, num_heads=32, max_len=256, dropout=0.1, causal=True, expansion_factor=1, device='cpu'):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        self.dropout = dropout
        self.causal = causal
        self.expansion_factor = expansion_factor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.causal = config.causal
        self.num_heads = config.num_heads
        self.scale = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("m", get_alibi_slope(self.num_heads).to(config.device))
        self.kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False).to(config.device)
        if config.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(1, 1, config.max_len, config.max_len)).to(config.device)
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        bias = (self.m * get_relative_positions(seq_len).to(x.device)).unsqueeze(0)

        score = torch.matmul(query, key) / self.scale + bias

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.dropout(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        d_hidden = config.d_model * config.expansion_factor
        self.fc1 = nn.Linear(config.d_model, d_hidden).to(config.device)
        self.fc2 = nn.Linear(d_hidden, config.d_model).to(config.device)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.gelu(self.fc1(x))
        out = self.dropout(self.fc2(x))
        return out

class ALiBiTransformerLayer(nn.Module):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.ffn_norm = nn.LayerNorm(config.d_model).to(config.device)
        self.attn_norm = nn.LayerNorm(config.d_model).to(config.device)
        self.ffn = FeedForward(config)
        self.attn = ALiBiMultiHeadAttention(config)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ALiBiTransformerEncoder(nn.Module):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ALiBiTransformerLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model).to(config.device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)