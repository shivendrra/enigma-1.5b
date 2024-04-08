import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from tokenizer import KMerTokenizer

with open('../parquet files/extra.txt', 'r', encoding='utf-8') as f:
  data = f.read()

print("file opened!")

model_prefix = '../tokenizer/trained models/base_2k.json'

tokenizer = KMerTokenizer(k=4)
tokenizer.build_vocab([data])
tokenizer.save_vocab(model_prefix)

# loaded_tokenizer = KMerTokenizer(k=4)
# loaded_tokenizer.load_vocab("../tokenizer/trained models/base_2k.json")

# with open('../parquet files/train.txt', 'r', encoding='utf-8') as f:
#   sample = f.read()

# print('encoding dna')
# encoded_tokens = tokenizer.encode_sequence(sample)
# decoded_tokens = tokenizer.decode_sequence(encoded_tokens)

# print(sample == decoded_tokens)
# print(f"sample length: {len(sample)}")
# print(f"tokens length: {len(encoded_tokens)}")
# print(f"file length: {len(sample)} \ntokens: {len(encoded_tokens)}")
# print(f"compression ration: {(len(sample) / len(encoded_tokens)):.2f}x")