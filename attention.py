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

    xq_ = torch.view_as_copmplex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis_broadcast = freqs_cis[..., :xq_.shape[-1]]

    rotated_xq = xq_ * freqs_cis_broadcast
    rotated_xk = xk_ * freqs_cis_broadcast

    xq_out = torch.view_as_real(rotated_xq).flatten(3)
    xk_out = torch.view_as_real(rotated_xk).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

freqs_cis = rope_calculation(head_dim, max_position_embeddings, base=rope_theta, device=hidden_states.device)

query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, freqs_cis)

class SimpleL2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
if use_qk_norm:
    qk_norm = SimpleL2Norm()
    query_states_final = qr_norm(query_states_rope)
    key_states_final = qr_norm(key_states_rope)
else:
    query_states_final = query_states_rope
    key_states_final = key_states_rope
    print("\nSkipped QK Norm")

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states= hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
key_states_final = repeat_kv(key_states_final, num_attention_heads // num_key_value_heads)

attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))

scaling_factor = 1.0 / math.sqrt(head_dim)
attn_weights = attn_weights * scaling_factor

if attention_mask is not None:
    casual_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]]
    attn_weights = attn_weights + casual_mask
else:
    print("\nNo attention mask applied")

attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
attn_output = torch.matmul(attn_weights, value_states_repeated)

attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.view(batch_size, sequence_length, hidden_size)

final_attn_output = o_proj(attn_output)

print("\nFinal Output Shapes:")
print(f"  attn_output (reshaped): {attn_output.shape}")
print(f"  final_attn_output: {final_attn_output.shape}")

class simplified_attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        self.num_key_value_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config['max_position_embeddings']
        self.rope_theta = config['rope_theta']
        self.attention_bias = config['attention_bias']
        self.use_qk_norm = config['use_qk_norm']

        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        self.freqs_cis = simple_rope_calculation(self.head_dim, self.max_position_embeddings, base=self.rope_theta)

        if self.use_qk_norm:
             self.qk_norm = SimpleL2Norm()

    def forward(self, hidden_states, attention_mask, position_ids):
        batch_size, sequence_length, _ = hidden_states.shape

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        current_freqs_cis = self.freqs_cis.to(hidden_states.device) # Get precomputed freqs
        query_states_rope, key_states_rope = apply_rotary_emb_torch(query_states, key_states, current_freqs_cis)

        # Optional QK Norm
        if self.use_qk_norm:
             query_states_final = self.qk_norm(query_states_rope)
             key_states_final = self.qk_norm(key_states_rope)
        else:
            query_states_final = query_states_rope
            key_states_final = key_states_rope


        # Repeat K/V for GQA
        key_states_repeated = repeat_kv(key_states_final, self.num_key_value_groups)
        value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)

        # Attention Calculation
        attn_weights = torch.matmul(query_states_final, key_states_repeated.transpose(2, 3))
        scaling_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = attn_weights * scaling_factor

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states_repeated.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        # Dropout would be here in training

        attn_output = torch.matmul(attn_weights, value_states_repeated)

        # Reshape and Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, self.hidden_size)
        final_attn_output = self.o_proj(attn_output)

        return final_attn_output, attn_weights # Return weights for inspection
    
config_dict = {
    'hidden_size': hidden_size,
    'num_attention_heads': num_attention_heads,
    'num_key_value_heads': num_key_value_heads,
    'max_position_embeddings': max_position_embeddings,
    'rope_theta': rope_theta,
    'attention_bias': attention_bias,
    'use_qk_norm': use_qk_norm,
}

simplified_attn_module = simplified_attention(config_dict)

# Run forward pass
final_output_simplified, final_weights_simplified = simplified_attn_module(hidden_states, attention_mask, position_ids)

print("\nOutput shape from simplified module:", final_output_simplified.shape)
print("Attention weights shape from simplified module:", final_weights_simplified.shape)
