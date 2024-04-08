import os
from tqdm import tqdm
import json
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class KMerTokenizer:
  def __init__(self, k_mers):
    self.k_mers = k_mers
    self.vocab = {}
    self.id_to_token = []
    self.token_to_id = {}

  def tokenize_sequence(self, sequence):
    kmers = [sequence[i:i+self.k_mers] for i in tqdm(range(0, len(sequence), self.k_mers), desc="tokenizing k-mers")]
    return kmers

  def build_vocab(self, sequences):
    all_kmers = []
    for sequence in sequences:
      all_kmers.extend(self.tokenize_sequence(sequence))
    token_count = {}
    for kmer in all_kmers:
      if kmer in token_count:
        token_count[kmer] += 1
      else:
        token_count[kmer] = 1
    sorted_tokens = sorted(token_count.items(), key=lambda x: x[1], reverse=True)
    for token, _ in sorted_tokens:
      self.token_to_id[token] = len(self.token_to_id)
      self.id_to_token.append(token)
    self.vocab = self.token_to_id

  def encode(self, sequence):
    encoded_sequence = []
    kmers = self.tokenize_sequence(sequence)
    for kmer in tqdm(kmers, desc="encoding sequences"):
      if kmer in self.token_to_id:
        encoded_sequence.append(self.token_to_id[kmer])
      else:
        encoded_sequence.append(len(self.vocab))
    return encoded_sequence

  def decode(self, encoded_sequence):
    decoded_sequence = [self.id_to_token[token_id] for token_id in encoded_sequence]
    return decoded_sequence
  
  def save_model(self, model_path):
    vocab_file = f"{model_path}/base_{self.k_mers}k.json"
    with open(vocab_file, 'w') as f:
      json.dump(self.vocab, f)

  def load_model(self, path):
    assert path.endswith('.json')
    with open(path, 'r') as f:
      vocab = json.load(f)
    
    self.vocab = vocab
    self.vocab_size = len(vocab)

with open('../parquet files/extra.txt', 'r', encoding='utf-8') as f:
  train_data = f.read()
  print("file opened!")

tokenizer = KMerTokenizer(k_mers=5)
tokenizer.build_vocab([train_data])
# tokenizer.save_model('trained models')
tokenizer.load_model('trained models/base_5k.json')

with open('../parquet files/train.txt', 'r', encoding='utf-8') as f:
  test_data = f.read()
  print("file opened!")

encoded_sequence = tokenizer.encode(test_data)
print(encoded_sequence[:100])
decoded_sequence = tokenizer.decode(encoded_sequence)
print("working fine:", decoded_sequence == test_data)