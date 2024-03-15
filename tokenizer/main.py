from tqdm import tqdm
import json

class DNAtokenizer:
  def __init__(self):
    super().__init__()
    self.chars = ["\n", "A", "C", "G", "T"]
    self.vocab_size = len(self.chars)
    self.merges = {}
    self.vocab = {}
    self.string_to_index = {char: idx for idx, char in enumerate(self.chars)}
    self.index_to_string = {idx: char for idx, char in enumerate(self.chars)}
  
  def _encode(self, string):
    encoded = [self.string_to_index[char] for char in string]
    return encoded
  
  def _decode(self, integer):
    decoded = ''.join([self.index_to_string[i] for i in integer])
    return decoded

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

  def _build_vocab(self):
    return {i: ids for i, ids in enumerate(self.chars)}

  def train(self, train_data, target_vocab):
    vocab = self._build_vocab()
    tokens = self._encode(train_data)
    ids = list(tokens)
    
    merges = {}
    n_merges = target_vocab - self.vocab_size + 1
    for i in tqdm(range(n_merges), desc='Training the tokenizer\t'):
      stats = self._get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = self.vocab_size + i
      ids = self._merge(ids, pair, idx)
      merges[pair] = idx
    
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    self.vocab = vocab
    self.merges = merges
    self.vocab_size = len(vocab)
  
  def continue_train(self, train_data, n_merges):
    tokens = self._encode(train_data)
    ids = list(tokens)
    for i in tqdm(range(n_merges), desc='Training continue'):
      stats = self._get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = self.vocab_size + i
      ids = self._merge(ids, pair, idx)
      self.merges[pair] = idx
    
    for (p0, p1), idx in self.merges.items():
      self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    self.vocab_size = len(self.vocab)
  
  def encode(self, text):
    tokens = self._encode(text)
    ids = list(tokens)
    while len(ids) >= 2:
      stats = self._get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break

      idx = self.merges[pair]
      ids = self._merge(ids, pair, idx)
    return ids

  def decode(self, de_text):
    tokens = [self.vocab[idx] for idx in de_text]
    text = ''.join(tokens)
    return text
  
  def save_model(self, model_prefix):
    model_file = model_prefix + '.model'
    with open(model_file, 'w', encoding='utf-8') as fwrite:
      for ids1, ids2 in self.merges:
        fwrite.write(f"{ids1} {ids2}\n")
    vocab_file = model_prefix + '_vocab.json'
    with open(vocab_file, 'w') as f:
      json.dump(self.vocab, f)
    print('model file saved successfully!')

  def load_model(self, model_path):
    assert model_path.endswith('.model')

    merges = {}
    idx = self.vocab_size
    with open(model_path, 'r', encoding='utf-8') as fread:
      for line in fread:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    vocab = self._build_vocab()

    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    self.merges = merges
    self.vocab = vocab
    self.vocab_size = len(self.vocab)