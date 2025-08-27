import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


hidden_size = 128
num_attention_heads = 16 # Dividing key into 16 parts
num_key_value_heads = 4 # Number of key/value heads
head_dim = hidden_size // num_attention_heads # Dimension of each attentionn head
max_position_embeddings = 256 # Maximum sequence length the model expects (usually it can handle way more)
rope_theta = 10000.0
rms_horm_eps = 1e-5 # Espsilon for RMSNorm
attention_bias = False
attentino_dropout = 0.0
use_qk_norm = True

batch_size = 2
sequence_length = 10
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)

position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1)

attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal=1)
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
attention_mask = attention_mask.expand(batch_size, 1, -1, -1)

q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

query_states = q_proj(hidden_states)
key_states = k_proj(hidden_states)
value_states = v_proj(hidden_states)

query_states = query_states.view(batch_size, sequence_length, num_attention_heads, head_dim).transpose(1, 2)
key_states= key_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)
value_states = value_states.view(batch_size, sequence_length, num_key_value_heads, head_dim).transpose(1, 2)

def rope_calculation(dim, max_seq_len, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arrange(0, dim, 2, device=device).float() / dim))
    t = torch.arrange(max_seq_len, device=device).type_as(inv_freq)
    freqs = new_func(inv_freq, t)
    emb = torch.cat((freqs, freqs), dim=-1)

    freqs_cos = emb.cos()
    freqs_sin = emb.sin()

    freqs_cis = torch.complex(freqs_cos, freqs_sin)
    return freqs_cis

def new_func(inv_freq, t):
    freqs= torch.outer(t, inv_freq)
    return freqs

def apply_rotary_emb_torch(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis.to(xq.device)
    freqs_cis = freqs_cis[position_ids]

    freqs_cis = freqs_cis[:, None, :, :]