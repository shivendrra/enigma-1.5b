import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

with open('new_dna.txt', 'r', encoding='utf-8') as file:
  captions = file.read()

print(f"{(len(captions)/1e6):.2f} million letters")

chars = sorted(list(set(captions)))
print(chars)
vocab_size = len(chars)

# map of characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import Transformer
model = Transformer(vocab_size=vocab_size)

checkpoint_path = '../trained models/enigma_47m.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
m = model.to(device)

target_text = "AGTTCTGCGAT"
context = torch.tensor([encode(target_text)], dtype=torch.long, device=device)
generated_output = decode(m.generate(context, max_new_tokens=10, temperature=0.5, top_k=5))
print(f"{target_text}{generated_output}")