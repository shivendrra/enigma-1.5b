import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

import json
import collections
from tqdm import tqdm

class KMerTokenizer:
    def __init__(self, k):
        self.k = k
        self.chars = ['\n', 'A', 'T', 'G', 'C', 'P', 'M', 'U', ' ']
        self.vocab = {"\n": 1, "A": 2, "T": 3, "G": 4, "C": 5, "P": 6, "M": 7, "U": 8, " ": 9}
        self.id_to_token = []
        self.token_to_id = {}

    def tokenize_sequence(self, sequence):
        kmers = [sequence[i:i+self.k] for i in range(0, len(sequence), self.k)]
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

    def train(self, corpus, max_merge_operations):
        for _ in tqdm(range(max_merge_operations), desc='Training the tokenizer\t'):
            token_counts = collections.Counter()
            for sequence in corpus:
                kmers = self.tokenize_sequence(sequence)
                token_counts.update(kmers)
            most_common_pair = max(token_counts.items(), key=lambda x: x[1])[0]
            self.vocab[''.join(most_common_pair)] = len(self.vocab)
            self._update_token_mappings()

    def encode_sequence(self, sequence):
        encoded_sequence = []
        kmers = self.tokenize_sequence(sequence)
        for kmer in kmers:
            if kmer in self.token_to_id:
                encoded_sequence.append(self.token_to_id[kmer])
            else:
                encoded_sequence.append(len(self.vocab))
        return encoded_sequence

    def decode_sequence(self, encoded_sequence):
        decoded_sequence = ''.join([self.id_to_token[token_id] for token_id in encoded_sequence])
        return decoded_sequence

    def _update_token_mappings(self):
        self.id_to_token = [token for token, _ in sorted(self.vocab.items(), key=lambda x: x[1])]
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}

    def save_vocab(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.vocab, file)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}