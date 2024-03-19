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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import Transformer
model = Transformer(vocab_size=vocab_size)

checkpoint_path = '../trained models/enigma_47m.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
m = model.to(device)

target_text = "AGTTCTGCGAT"
context = torch.tensor([tokenizer.encode(target_text)], dtype=torch.long, device=device)
generated_output = tokenizer.decode(m.generate(context, max_new_tokens=10, temperature=0.5, top_k=5))
print(f"{target_text}{generated_output}")