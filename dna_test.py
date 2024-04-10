import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from tokenizer import KMerTokenizer

# with open('../parquet files/train1.txt', 'r', encoding='utf-8') as f:
#   train_data = f.read()
#   print("file opened!")

tokenizer = KMerTokenizer(k_mers=5)
# tokenizer.build_vocab([train_data])
# tokenizer.save_model('../tokenizer/trained models')

with open('../parquet files/train1.txt', 'r', encoding='utf-8') as f:
  test_data = f.read()
  print("file opened!")

tokenizer.load_model('../tokenizer/trained models/base_5k.json')

encoded_tokens = tokenizer.encode(test_data)
decoded_tokens = tokenizer.decode(encoded_tokens)
print(decoded_tokens)

print(f"seq length: {len(test_data)} \ntokens length: {len(decoded_tokens)}")
print(test_data == decoded_tokens)
print(f"file length: {len(test_data)} \ntokens: {len(encoded_tokens)}")
print(f"compression ration: {(len(test_data) / len(encoded_tokens)):.2f}x")