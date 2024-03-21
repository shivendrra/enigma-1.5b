import numpy as np
from tqdm import tqdm
import json
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class NewTokenizer:
    def __init__(self):
        """
        Initialize tokenizer attributes.
        """
        self.chars = ["\n", "A", "C", "G", "T", " "]
        self.vocab_size = len(self.chars)
        self.merges = {}
        self.vocab = {}
        self.string_to_index = {char: idx for idx, char in enumerate(self.chars)}
        self.index_to_string = {idx: char for idx, char in enumerate(self.chars)}

    def _encode(self, string):
        """
        Encode a string into a list of integers.
        """
        encoded = np.array([self.string_to_index[char] for char in string], dtype=np.int32)
        return encoded

    def _merge(self, ids, pair, idx):
        """
        Replace all consecutive pair occurrences in ids with idx.
        """
        mask = np.logical_and(ids[:-1] == pair[0], ids[1:] == pair[1])
        new_ids = np.where(mask, idx, ids)
        return new_ids[new_ids != pair[1]]  # Remove pair[1] from new_ids

    def _build_vocab(self):
        """
        Build the initial vocabulary.
        """
        return {i: ids for i, ids in enumerate(self.chars)}

    def train(self, train_data, target_vocab):
        """
        Train the tokenizer.
        """
        vocab = self._build_vocab()
        tokens = self._encode(train_data)
        ids = list(tokens)
        
        merges = {}
        n_merges = target_vocab - self.vocab_size + 1
        for i in tqdm(range(n_merges), desc='Training the tokenizer\t'):
            pair_counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.int32)
            for id1, id2 in zip(ids, ids[1:]):
                pair_counts[id1, id2] += 1
            max_pair = np.unravel_index(np.argmax(pair_counts), pair_counts.shape)
            idx = self.vocab_size + i
            ids = self._merge(ids, max_pair, idx)
            merges[max_pair] = idx

        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        self.vocab = vocab
        self.merges = merges
        self.vocab_size = len(vocab)

    def encode(self, text):
        """
        Encode input text using trained tokenizer.
        """
        tokens = self._encode(text)
        ids = tokens.tolist()
        while len(ids) >= 2:
            pair_counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.int32)
            for id1, id2 in zip(ids, ids[1:]):
                pair_counts[id1, id2] += 1
            max_pair = tuple(np.unravel_index(np.argmax(pair_counts), pair_counts.shape))
            if max_pair not in self.merges:
                break
            idx = self.merges[max_pair]
            ids = self._merge(ids, max_pair, idx).tolist()
        return ids

    def decode(self, de_text):
        """
        Decode list of integers into a string using the vocabulary.
        """
        tokens = [self.vocab[idx] for idx in de_text]
        text = ''.join(tokens)
        return text

    def save_model(self, model_prefix):
        """
        Save model and vocab to files.
        """
        model_file = model_prefix + '.model'
        with open(model_file, 'w', encoding='utf-8') as fwrite:
            for (ids1, ids2), idx in self.merges.items():
                fwrite.write(f"{ids1} {ids2} {idx}\n")
        vocab_file = model_prefix + '_vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
        print('Model files saved successfully!')

    def load_model(self, model_path):
        """
        Load model from files.
        """
        assert model_path.endswith('.model')
        merges = {}
        vocab = self._build_vocab()
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
