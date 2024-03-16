import timeit
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from tokenizer import DNAtokenizer

start = timeit.default_timer()
with open('../parquet files/train.txt', 'r', encoding='utf-8') as f:
  data = f.read()

start_token = timeit.default_timer()
print(f"file opened in {((start_token - start)/60):.2f} mins")

model_prefix = '../tokenizer/trained models/base_2k'
os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

token = DNAtokenizer()
# token.train(data, 250)

token.load_model(model_path='../tokenizer/trained models/base_1k.model')
# token.continue_train(data, 7)
# token.save_model(model_prefix=model_prefix)

end_token = timeit.default_timer()
print(f"tokenized in {((end_token - start_token)/60):.2f} mins")

# sample = 'CCTCCTGCCTGGAACATCAGGCTCCATGTTCTTTGGCTTTTAGAC'
with open('../enigma/new_dna.txt', 'r', encoding='utf-8') as f:
  sample = f.read()

# print(token.encode(sample))
encoded_tokens = token.encode(sample)
decoded_tokens = token.decode(encoded_tokens)
print(sample == decoded_tokens)
print(f"file length: {len(sample)} \ntokens: {len(encoded_tokens)}")
print(f"compression ration: {(len(sample) / len(encoded_tokens)):.2f}x")