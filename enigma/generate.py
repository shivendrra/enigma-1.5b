import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

with open('../parquet files/new_dna.txt', 'r', encoding='utf-8') as file:
  captions = file.read()

print(f"{(len(captions)/1e6):.2f} million letters")

from tokenizer import PerCharTokenizer

tokenizer = PerCharTokenizer()
vocab_size = tokenizer.vocab_size

import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import Transformer
model = Transformer(vocab_size=vocab_size)

class Generator(Transformer):
  def __init__(self, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.block_size = Transformer.block_size

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
      idx_cond = idx[:, -self.block_size:]
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
  
  def generate_masked_tokens(self, idx, masked_indices, temperature=1.0, top_k=0):
    """
      Generate predictions for masked tokens using the trained model.

      Args:
        - idx (Tensor): input tensor representing token indices
        - masked_indices (Tensor): tensor of indices indicating masked positions
        - temperature (float): softmax temperature for sampling
        - top_k (int): no of top tokens to consider in sampling

      Returns:
        - predicted_tokens (Tensor): tensor of predicted token indices
    """
    B, T = idx.shape

    toked_model = self.toked_model(idx)
    pos_encod = self.pos_encod(torch.arange(T, device=device))
    x = toked_model + pos_encod

    for layer in self.enc_layer:
      x_out = layer(x)

    for layer in self.dec_layer:
      x_final = layer(x, x_out)

    x_masked = x_final.clone()
    x_masked[masked_indices] = self.toked_model(torch.tensor([6], device=device))

    x_masked = self.norm_final(x_masked)
    logits = self.linear_final(x_masked)

    masked_logits = logits[masked_indices].view(-1, logits.size(-1))
    scaled_logits = masked_logits / temperature
    if top_k > 0:
      scaled_logits = self._top_k_filtering(scaled_logits, top_k)

    probs = F.softmax(scaled_logits, dim=-1)
    predicted_indices = torch.argmax(probs, dim=-1)

    return predicted_indices
  
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

checkpoint_path = '../trained models/enigma_47m.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
m = model.to(device)

target_text = "AGTTCTGCGAT"
context = torch.tensor([tokenizer.encode(target_text)], dtype=torch.long, device=device)
generated_output = tokenizer.decode(Generator.generate(context, max_new_tokens=10, temperature=0.5, top_k=5))
print(f"{target_text}{generated_output}")