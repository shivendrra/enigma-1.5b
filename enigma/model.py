"""
  transformer based model, but with few minimal tweaks
  trained a 150million parameters model with current set configurations
"""

import torch
import json
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

import torch.nn as nn
from torch.nn import functional as F

with open('config_enigma.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

batch_size = params['batch_size']
block_size = params['block_size']
n_head = params['n_head']
d_model = params['d_model']
n_layers = params['n_layer']
dropout = params['dropout']
norm_eps = params['norm_eps']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionHead(nn.Module):
  """
    initialize a single head of self attention.

    Args:
    - d_model (int): dimensionality of the model's hidden layers
    - head_size (int): dimensionality of each attention head
    - dropout (float): dropout probability
    - block_size (int): the maximum sequence length for positional encoding
  """
  def __init__(self, d_model, head_size, dropout, block_size):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x, mask=False):
    """
    forward pass of a single attention head.

    Args:
      - x (Tensor): input tensor.
      - mask (bool): flag indicating whether to apply masking

    Returns:
      - out (Tensor): output tensor after self attention
    """
    B, T, C = x.shape
    key = self.key(x)
    query = self.query(x)

    weights = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1]**-0.5)

    if mask is True:
      weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

    weights = F.softmax(weights, dim=-1)
    weights = self.dropout(weights)

    value = self.value(x)
    out = torch.matmul(weights, value)
    return out

class MultiHeadAttention(nn.Module):
  """
    initialize a multi-head attention module.

    Args:
    - d_model (int): dimensionality of the model's hidden layers
    - n_head (int): no of attention heads
    - dropout (float): dropout probability
    - block_size (int): context length
  """
  def __init__(self, d_model, n_head, dropout, block_size):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(d_model=d_model, dropout=dropout, head_size=head_size, block_size=block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    """
    forward pass of the multi-head attention module

    Args:
      - x (Tensor): input tensor
      - mask (bool): flag indicating whether to apply masking

    Returns:
      - out (Tensor): output tensor after multi-head attention

    """
    out = torch.cat([h(x, mask=mask) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  """
    initialize a feedforward network module

    Args:
    - d_model (int): the dimensionality of the model's hidden layers
    - dropout (float): dropout probability

  """
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 5*d_model),
      nn.GELU(),
      nn.Linear(5*d_model, d_model),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    """
    forward pass of the feedforward network module

    Args:
      - x (Tensor): input tensor

    Returns:
      - out (Tensor): output tensor after passing through the feedforward network
    """
    return self.net(x)

class EncoderNetwork(nn.Module):
  """
    initialize an encoder network module

    Args:
    - d_model (int): dimensionality of the model's hidden layers
    - n_head (int): no of attention heads in multi-head attention layers
    - norm_eps (float): epsilon value for layer normalization
    - dropout (float): dropout probability
    - block_size (int): the maximum sequence length for positional encoding
    """
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.s_att = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

  def forward(self, src):
    """
      forward pass of the encoder network module.
    
      Args:
      - src (Tensor): input tensor representing source data

      Returns:
      - src (Tensor): output tensor after passing through the encoder network
    """
    src2 = self.s_att(src, mask=False)
    src = src + self.dropout(src2)
    src = self.norm1(src)

    src2 = self.ffwd(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)

    return src

class DecoderNetwork(nn.Module):
  """
    initialize a decoder network module

    Args:
      - d_model (int): dimensionality of the model's hidden layers
      - n_head (int): no of attention heads in multi-head attention layers
      - norm_eps (float): epsilon value for layer normalization
      - dropout (float): dropout probability
      - block_size (int): the maximum sequence length for positional encoding
  """
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.s_att = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

  def forward(self, src, trg):
    """
      forward pass of the decoder network module.

      Args:
        - src (Tensor): input tensor representing source data
        - trg (Tensor): input tensor representing target data

      Returns:
        - src_f (Tensor): output tensor after passing through the decoder network
    """
    src2 = self.s_att(src, mask=True)
    src = src + self.dropout(src2)
    src = src + self.norm1(src)

    trg2 = self.s_att(trg, mask=False)
    trg = trg + self.dropout(trg2)
    trg = trg + self.norm1(trg)

    src_f = src + trg
    src_f2 = self.ffwd(self.norm2(src_f))
    src_f = src_f + self.dropout(src_f2)
    src_f = self.norm2(src_f)

    return src_f

class Transformer(nn.Module):
  """
    initialize a Transformer model

    Args:
      - vocab_size (int): size of the vocabulary
      - d_model (int): dimensionality of the model's hidden layers
      - block_size (int): maximum sequence length for positional encoding/context length
      - n_layers (int): number of encoder and decoder layers in the Transformer
      - n_head (int): number of attention heads in multi-head attention layers
      - norm_eps (float): epsilon value for layer normalization
      - dropout (float): dropout probability
  """
  def __init__(self, vocab_size):
    super().__init__()
    self.toked_model = nn.Embedding(vocab_size, d_model)
    self.pos_encod = nn.Embedding(block_size, d_model)
    self.enc_layer = nn.ModuleList([EncoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])
    self.dec_layer = nn.ModuleList([DecoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])

    self.norm_final = nn.LayerNorm(d_model)
    self.linear_final = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """
      initialize weights of linear and embedding layers

      Args:
        - module (nn.Module): the module to initialize weights for
    """
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    """
      forward pass of the transformer model

    Args:
      - idx (Tensor): input tensor representing token indices
      - targets (Tensor): target tensor for computing loss during training

    Returns:
      - logits (Tensor): output logits from the final linear layer
      - loss (Tensor): optional. computed cross-entropy loss if targets are provided, else None
    """
    B, T = idx.shape

    toked_model = self.toked_model(idx)
    pos_encod = self.pos_encod(torch.arange(T, device=device))
    x = toked_model + pos_encod

    for layer in self.enc_layer:
      x_out = layer(x)

    for layer in self.dec_layer:
      x_final = layer(x, x_out)

    x_final = self.norm_final(x_final)
    logits = self.linear_final(x_final)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
    """
      generate new tokens using the trained model

    Args:
      - idx (Tensor): input tensor representing initial token indices
      - max_new_tokens (int): max no of new tokens to generate
      - temperature (float): softmax temperature for sampling
      - top_k (int): no of top tokens to consider in sampling

    Returns:
      - generated_tokens (list): list of generated token indices
    """
    generated_tokens = []

    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]

      scaled_logits = logits / temperature
      if top_k > 0:
        scaled_logits = self._top_k_filtering(scaled_logits, top_k)

      probs = F.softmax(scaled_logits, dim=-1)
      sampled_idx = torch.multinomial(probs, num_samples=1)
      generated_tokens.append(sampled_idx.item())
      idx = torch.cat((idx, sampled_idx), dim=1)

    return generated_tokens

  def _top_k_filtering(self, logits, top_k):
    """
      filter logits to keep only the top-k tokens

    Args:
      - logits (Tensor): input tensor representing unscaled logits
      - top_k (int): no of top tokens to keep

    Returns:
      - filtered_logits (Tensor): filtered logits with only top-k tokens remaining
    """
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1).expand_as(logits)
    filtered_logits = torch.where(logits < min_value, torch.ones_like(logits) * -float('inf'), logits)

    return filtered_logits