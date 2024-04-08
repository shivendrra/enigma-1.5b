import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from tqdm import tqdm
import json

class KmerPairTokenizer:
  def __init__(self):
    self.k_mers = 4
    self.vocab = {}
    self.merges = {}
    self.vocab_size = 0
    self.init_vocab = {"\n": 1, "A": 2, "T": 3, "G": 4, "C": 5, "P": 6, "M": 7, "U": 8, " ": 9}
  
  def _tokenize_seq(self, sequence):
    kmers = [sequence[i:i+self.k_mers] for i in tqdm(range(0, len(sequence), self.k_mers), desc="tokenizing k-mers")]
    return kmers
  
  def _get_stats(self, ids, counts=None):
    """
      takes list of integers and returns dictionary of counts of pairs(consecutive ones)
      eg: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
      allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
      counts[pair] = counts.get(pair, 0) + 1
    return counts

  def _merge(self, ids, pair, idx):
    """
      in the list of integers, replaces all consecutive pair with the new integer token idx
      eg: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
      if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
        new_ids.append(idx)
        i += 2
      else:
        new_ids.append(ids[i])
        i += 1
    return new_ids
  
  def get_ids(self, data):
    all_kmers = []
    seq_to_no = {}
    ass_no = []
    i = 1
    for seq in data:
      all_kmers.extend(self._tokenize_seq(seq))

    for seq in all_kmers:
      if seq not in seq_to_no:
        seq_to_no[seq] = i
        i += 1
      ass_no.append(seq_to_no[seq])
    
    del all_kmers, i
    return ass_no, seq_to_no

  def train_tokenizer(self, data: str, max_vocab: int):
    n_merges = max_vocab
    text_pairs, init_vocab = self.get_ids([data])
    ids = list(text_pairs)

    del text_pairs, max_vocab
    merges = {}
    ids_len = len(init_vocab)

    for i in tqdm(range(n_merges), desc="training the tokenizer"):
      stats = self._get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = ids_len + i + 1
      ids = self._merge(ids, pair, idx)
      merges[pair] = idx

    vocab = {value: key for key, value in init_vocab.items()}    
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]

    self.vocab = vocab
    self.merges = merges
    self.vocab_size = len(self.vocab)

    del vocab, merges, ids, stats, pair, idx
  
  def encode(self, text):
    text_pairs, _ = self.get_ids([text])
    ids = list(text_pairs)
    total_pairs = len(ids) - 1

    with tqdm(total=total_pairs, desc="Encoding text") as pbar:
      while len(ids) >= 2:
        stats = self._get_stats(ids)
        pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
        if pair not in self.merges:
          break
        idx = self.merges[pair]
        ids = self._merge(ids, pair, idx)
        pbar.update(1)
    return ids

  def decode(self, ids):
    tokens = [self.vocab[idx] for idx in ids]
    sequence = ''.join(tokens)
    return sequence
  
  def save_model(self, file_path):
    model_file = file_path + f"/base_mer.model"
    vocab_file = file_path + f"/base_kmer.json"

    with open(model_file, 'w', encoding='utf-8') as f:
      for ids1, ids2 in self.merges:
        f.write(f"{ids1} {ids2}\n")
    with open(vocab_file, 'w') as f:
      json.dump(self.vocab, f)
    print('model file saved successfully!')
  
  def load(self, model_path, vocab_path):
    assert model_path.endswith('.model')
    assert vocab_path.endswith('.json')

    with open(vocab_path, 'r') as f:
      vocab_data = json.load(f)
      
    self.vocab = vocab_data
    self.vocab_size = len(vocab_data)

    merges = {}
    idx = 256 + 1
    with open(model_path, 'r', encoding='utf-8') as fread:
      for line in fread:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    self.merges = merges
