# # import timeit
# # import os
# # current_directory = os.path.dirname(os.path.abspath(__file__))
# # os.chdir(current_directory)
# # from concurrent.futures import ThreadPoolExecutor
# # from tokenizer import DNAtokenizer

# # def train_batch(token, batch_data, target_vocab):
# #   token.train(batch_data, target_vocab)

# # def train_tokenizer_with_parallel_batches(input_file, batch_size, target_vocab, model_prefix):
# #   start = timeit.default_timer()
# #   with open(input_file, 'r', encoding='utf-8') as f:
# #       data = f.read()
# #   start_token = timeit.default_timer()
# #   print(f"File opened in {(start_token - start) / 60:.2f} mins")

# #   token = DNAtokenizer()
# #   total_batches = (len(data) + batch_size - 1) // batch_size
# #   print("total batches:", total_batches)
# #   batches = []
# #   for batch_idx in range(total_batches):
# #     start_batch = batch_idx * batch_size
# #     end_batch = min((batch_idx + 1) * batch_size, len(data))
# #     batch_data = data[start_batch:end_batch]
# #     batches.append(batch_data)

# #   with ThreadPoolExecutor() as executor:
# #     executor.map(lambda batch: train_batch(token, batch, target_vocab), batches)

  
  # os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
# #   token.save_model(model_prefix)

# #   end_token = timeit.default_timer()
# #   print(f"Tokenized in {(end_token - start_token) / 3600:.2f} hrs")

# #   return token

# # # Define paths and parameters
# # # input_file = 'parquet files/train1.txt'
# # input_file = 'enigma/new_dna.txt'
# # batch_size = 1000
# # target_vocab = 100
# # model_prefix = 'tokenizer/trained_models/base_1k'

# # # Train the tokenizer with parallel processing
# # trained_tokenizer = train_tokenizer_with_parallel_batches(input_file, batch_size, target_vocab, model_prefix)

# # # Test the tokenizer with a sample sequence
# # sample_sequence = 'CCTCCTGCCTGGAACATCAGGCTCCATGTTCTTTGGCTTTTAGAC'
# # encoded_sequence = trained_tokenizer.encode(sample_sequence)
# # decoded_sequence = trained_tokenizer.decode(encoded_sequence)
# # print(f"Sample sequence matches original: {sample_sequence == decoded_sequence}")
# # print(f"Compression ratio: {len(sample_sequence) / len(encoded_sequence):.2f}x")
# # print(f"{encoded_sequence}")

# import timeit
# import os
# current_directory = os.path.dirname(os.path.abspath(__file__))
# os.chdir(current_directory)
# from tokenizer import DNAtokenizer

# def train_tokenizer_with_batches(input_file, batch_size, target_vocab, model_prefix):
#     start = timeit.default_timer()

#     # Read the entire input data
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = f.read()

#     start_token = timeit.default_timer()
#     print(f"File opened in {(start_token - start) / 60:.2f} mins")

#     token = DNAtokenizer()

#     # Split the input data into batches and train the tokenizer on each batch
#     total_batches = (len(data) + batch_size - 1) // batch_size
#     for batch_idx in range(total_batches):
#         start_batch = batch_idx * batch_size
#         end_batch = min((batch_idx + 1) * batch_size, len(data))
#         batch_data = data[start_batch:end_batch]

#         print(f"Training on batch {batch_idx + 1}/{total_batches}")
#         token.train(batch_data, target_vocab)

#     os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
#     token.save_model(model_prefix)

#     end_token = timeit.default_timer()
#     print(f"Tokenized in {(end_token - start_token) / 60:.2f} mins")

#     return token

# # Define paths and parameters
# input_file = 'enigma/new_dna.txt'
# batch_size = 1000  # Adjust batch size according to memory constraints
# target_vocab = 100
# model_prefix = 'tokenizer/trained_models/base_1k'

# # Train the tokenizer with batch processing
# trained_tokenizer = train_tokenizer_with_batches(input_file, batch_size, target_vocab, model_prefix)

# # Test the tokenizer with a sample sequence
# sample_sequence = 'CCTCCTGCCTGGAACATCAGGCTCCATGTTCTTTGGCTTTTAGAC'
# encoded_sequence = trained_tokenizer.encode(sample_sequence)
# decoded_sequence = trained_tokenizer.decode(encoded_sequence)
# print(f"Sample sequence matches original: {sample_sequence == decoded_sequence}")
# print(f"Compression ratio: {len(sample_sequence) / len(encoded_sequence):.2f}x")
