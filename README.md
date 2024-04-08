# Enigma-1.5b
Using text-based seq-2-seq transformer to generate new DNA sequences. Very basic in nature, coz I don't have any domain expertise in dna generation. I've made two models, on is 47million parameters model and other is 2.5billion parameters.
Check out the model on huggingface: [enigma-1.5b](https://huggingface.co/Shivendrra/enigma-1.5b)

## Overview
It's a 2.5b model trained on ~1billion individual letters of DNA, kinda like training a text-based model on per-character level instead of sub-word level. It does have it's own tokenizer similar that is intersection b/w char-level and bpe-tokenizer.
For EnBERT i.e. decoder-only model is trained on lot's of sequences of DNA tokenized using `k-mer` tokenizer specially trained for this purpose, which means it has a larger vocab size than the enigma-2.5b.

## How to use:
Follow these steps to train your own tokenizer or generate outputs from the trained model:
1. Clone this repository:
	```shell
	git clone https://github.com/shivendrra/enigma-1.5b
	cd enigma-clone
	```

2. Install Dependencies:
	```shell
	pip install requirements.txt
	```

3. Train:
	1. Download all the datasets from the HuggingFace model, I've uploaded the data.
	2. Use the Google Colab notebooks with the set hyper-parameters.
	3. Train and Save the model. Have fun!!

## Model Architecture:
![architecture.png](https://github.com/shivendrra/enigma-1.5b/blob/main/architecture.png)

EnBERT is a 47million model, follows decoder-only architecture, whereas Engima-2.5b is a transformer model. It has a fairly complex model.
### Highlights
---
1. **Positional Embeddings:** Enigma model has positional embeddings added to the token embeddings in the start of all the processing, on the other hand, decoder-only model doesn't uses positional encodings in the start, it applies it in attention head to each attention matrix individually.
2. **RMS Normalization & Pre-normalization:** Both of the model uses RMS normalization same as implemented in LLaMa-2 and uses pre-normalization for model's stability while training.
3. **Self-Attention Layer:** Single headed attention layers have relative positional embeddings added to the attention matrix before masking. Masking is only done to first attention layer. Key, Query and Value matrices have biases added to them. `MultiHeadAttention` Layer concatenates all the outputs from each of the attention heads together.
4. **FeedForward:** Basic feed-forward network that has two linear layers with expansion factor of 5. GELU is used as activation function for this model instead of ReLU.
5. **Generation:** Token generation is very simple, passes context-tokens into the model, gets processed and then it just gives out some probabilities of values which are filtered using `argmax` function.

## Tokenizer:
It uses blend of basic character level tokenization and bpe-tokenization process. Initial vocab is defined already. 'P', 'M', 'U' are padding, mask and unknown tokens representation, respectively.
```python
chars = ['\n', 'A', 'T', 'G', 'C', 'P', 'M', 'U', ' ']
```
Then it uses the same process as bpe to build new vocab by merging max pairs in each iterations. This tokenizer is trained till 1.5k vocab size, already, and it can be loaded for later use.

*One issue that I've encountered is that it is pretty slow while tokenizing files, that's why it isn't used in the model while training.*

Here is how you can use it:
```python
from tokenizer import DNAtokenizer
token = DNAtokenizer()

# for training tokenizer
token.train(data, 250)
token.save_model(model_prefix=model_prefix)

# for loading save vocab
token.load_model(model_path='../tokenizer/trained models/base_1k.model')

sample = "ATTGCTA"
encoded_tokens = token.encode(sample)
decoded_tokens = token.decode(encoded_tokens)
```

## How it works?
Let's take a sample input of DNA seq: `"AGTTCTGCGAT"`, we feed it into the train model, and it will generate the next few letters of DNA, limit is 256 for now. Generate function uses `top_k` sampling and `temperature` setting. Use `enigma/generate.py` to generate outputs from the model.

anyways, here's a sample code to generate outputs from the model:
```python
from model import Transformer

model = Transformer(vocab_size=vocab_size)
checkpoint_path = '../trained models/enigma_47m.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
m = model.to(device)

target_text = "AGTTCTGCGAT"
context = torch.tensor([tokenizer.encode(target_text)], dtype=torch.long, device=device)
generated_output = tokenizer.decode(m.generate(context, max_new_tokens=10, temperature=0.5, top_k=5))
print(f"{target_text}{generated_output}")
```

## Training Summary:
These models were trained to 3k-4k iterations, each. on ~500million letters of DNA, roughly around 500mbs of data. Final losses were around ~0.02 for 47million parameter model and ~0.003 for 2.5billion parameter model. I had saved more data, lot more than this, but couldn't train it more due to technical in-capabilities.
Try to train it yourself if possible. `enigma/TrainEnigma` file contains all necessary functions needed to train it, from scratch or pre-train.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
MIT License