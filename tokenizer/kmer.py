from tqdm import tqdm
import json

class KMerTokenizer:
  def __init__(self, k_mers: int=4):
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
        break
    return encoded_sequence

  def decode(self, encoded_sequence):
    decoded_tokens = []
    for token_id in encoded_sequence:
      if token_id < len(self.id_to_token):
        decoded_tokens.append(self.id_to_token[token_id])
      else:
        break
    return ''.join(decoded_tokens)
  
  def save_model(self, model_path):
    vocab_file = f"{model_path}/base_{self.k_mers}k.json"
    with open(vocab_file, 'w') as f:
      json.dump(self.vocab, f)
    print("saved the vocab!")

  def load_model(self, path):
    assert path.endswith('.json')
    with open(path, 'r') as f:
      vocab = json.load(f)
    print("loaded the vocab!")
    
    self.vocab = vocab
    self.token_to_id = self.vocab
    self.vocab_size = len(vocab)

    self.id_to_token = [None] * self.vocab_size
    for token, idx in self.vocab.items():
        self.id_to_token[idx] = token