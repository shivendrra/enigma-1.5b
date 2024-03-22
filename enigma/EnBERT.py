"""
  simple BERT architecture model, paired with one more layer of 
  masked self-attention, to predict next token
"""

import torch
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparams
batch_size = 8
block_size = 32
max_iters = 10
eval_interval = 10
learning_rate = 3e-4
eval_iters = 5
d_model = 256
n_layer = 16
n_head = 12
dropout = 0.2
norm_eps = 1e-5

class SWiGLU(nn.Module):
  """ SWiGLU(x) = σ(x) ⊙ ReLU(x) + (1−σ(x)) ⊙ x """

  def forward(self, x):
    sigmoid_output = torch.sigmoid(x)
    relu_output = F.relu(x)
    out = sigmoid_output * relu_output + (1 - sigmoid_output) * x
    
    return out

class UnMaskedHead(nn.Module):
  """ single head of self attention """
  def __init__(self, d_model, head_size, dropout):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True) 
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    B, T, C = x.shape
    key = self.key(x)
    query = self.query(x)

    weights = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5
    weights = F.softmax(weights, dim=-1)
    weights = self.dropout(weights)

    value = self.value(x)
    out = weights @ value
    return out

class MaskedHead(nn.Module):
  """ one head of self-attention """
  def __init__(self, head_size, dropout, d_model):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    
    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    
    v = self.value(x)
    out = wei @ v
    return out
  
class MultiUnMasked(nn.Module):
  def __init__(self, d_model, n_head, dropout):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([UnMaskedHead(d_model=d_model, dropout=dropout, head_size=head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class MultiMasked(nn.Module):
  def __init__(self, d_model, n_head, dropout):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([MaskedHead(d_model=d_model, dropout=dropout, head_size=head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 4*d_model),
      nn.GELU(),
      nn.Linear(4*d_model, d_model),
      nn.Dropout(dropout)
    )
   
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, d_model, n_head, norm_eps, dropout):
    super().__init__()
    self.sa_masked = MultiMasked(n_head=n_head, d_model=d_model, dropout=dropout)
    self.sa_unmasked = MultiUnMasked(n_head=n_head, d_model=d_model, dropout=dropout)
    self.ffwd = FeedForward(d_model, dropout=dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
  
  def forward(self, x):
    x2 = x + self.sa_unmasked(self.norm1(x))
    x = x2 + self.norm2(self.ffwd(x2))

    x2 = x + self.sa_masked(self.norm1(x))
    x = x2 + self.norm2(self.ffwd(x2))
    return x

class EnigmaBERT(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.toked_model = nn.Embedding(vocab_size, d_model)
    self.pos_encod = nn.Embedding(block_size, d_model)
    self.block = nn.Sequential(*[Block(d_model=d_model, dropout=dropout, norm_eps=norm_eps, n_head=n_head) for _ in range(n_layer)])
    self.norm_final = nn.LayerNorm(d_model, eps=norm_eps)
    self.linear_final = nn.Linear(d_model, vocab_size)
    self.apply(self._init_weights)
  

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
  def forward(self, idx, targets=None):
    B, T = idx.shape

    toked_model = self.toked_model(idx)
    pos_encod = self.pos_encod(torch.arange(T, device=device))
    x = toked_model + pos_encod
    x = self.block(x)
    x = self.norm_final(x)
    logits = self.linear_final(x)

    if targets is None:
      loss = None
    
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
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
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1).expand_as(logits)
    filtered_logits = torch.where(logits < min_value, torch.ones_like(logits) * -float('inf'), logits)
    return filtered_logits