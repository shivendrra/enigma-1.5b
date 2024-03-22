# enigma-1.5b

Using text-based seq-2-seq transformer to generate new DNA sequences. Very basic in nature, coz I don't have any domain expertise in dna generation. I've made two models, on is 47million parameters model and other is 2.5billion parameters.
Check out the model on huggingface: [enigma-1.5b](https://huggingface.co/Shivendrra/enigma-1.5b)
## Overview
It's a 2.5 parameter model trained on ~1billion individual letters of DNA, kinda like training a text-based model on per-character level instead of sub-word level. It does have it's own tokenizer similar that is intersection b/w char-level and bpe-tokenizer.
## Model Architecture:

![[architecture.png](https://github.com/shivendrra/enigma-1.5b/blob/main/architecture.png)]

EnBERT is a 47million parameter model, follows BERT architecture, and has one more layer of masked self-attention layer to predict next tokens.
Engima-2.5b is a transformer model. It has a fairly complex model.

#### Encoder Part:
It consists two different layers, each followed by their own normalization and dropout layers. Input embeddings along with positional embeddings are fed to the encoder block:
##### Self Attention:
- Each head of self-attention layer is similar to that of used in `grokAI`. Key and Query matrices have biases whereas Value matrix doesn't.
- After implementing `torch.matmul()` on Key and Query, relational positional embeddings are applied to the attention matrix.
- Attention and value matrix are then multiplied using `torch.matmul()`.
- Multi-head attention layer than concatenates all the outputs together and passes them through a linear layer

#### FeedForward:
- Normalized outputs are then passed to position-wise `feedforward` layer, with `expansion_factor` of 5. 
- GELU is used as the activation function in this case and two linear layers, one for output and other for input.
- Finally dropout is applied and the outputs that are produced have deep global contextual information about the input tokens.
#### Decoder Part:
Consists of three different layers:
##### Masked Attention:
- This layer is similar to the self-attention implemented in encoder part, except it has a triangular mask that forbids tokens to look for the context of next token.
- Rest is all same, relational positional embeddings are applied in the same way, but to the masked attention matrix this time.
- Attention and value matrix are then multiplied using `torch.matmul()`.
- Multi-head attention layer than concatenates all the outputs together and passes them through a linear layer
#### Self-Attention:
- Before this, outputs from encoder layer and masked-attention layer are added together, and then passed to this layer.
- Same as the encoder's unmasked attention layer. Key, Query and Value matrices are created using same technique.
- Finally all the outputs are normalized and passed to dropout layer.

#### FeedForward:
- Normalized outputs are then passed to position-wise `feedforward` layer, with `expansion_factor` of 5. 
- GELU is used as the activation function in this case and two linear layers, one for output and other for input.
- Finally dropout is applied and the outputs that are produced have deep global contextual information about the input tokens.

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
### How it works?
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

## Training Part:
### Summary
These models were trained to 3k-4k iterations, each. on ~500million letters of DNA, roughly around 500mbs of data. Final losses were around ~0.02 for 47million parameter model and ~0.003 for 2.5billion parameter model. I had saved more data, lot more than this, but couldn't train it more due to technical in-capabilities.
Try to train it yourself if possible. `enigma/TrainEnigma` file contains all necessary functions needed to train it, from scratch or pre-train.
#### Functions:
This used a basic training procedure. `get_batch()` generated batches of data, `estimate_loss()` estimates losses and `train()` function is kind of master function, here, calling other functions after each or set iterations.

```python
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

from model import Transformer
model = Transformer(vocab_size=vocab_size)
m = model.to(device)

n_param = sum(p.numel() for p in m.parameters())/1e6
print(f"{n_param:.2f} million")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
steps = []
train_losses = []
val_losses = []

for iter in range(max_iters):
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    steps.append(iter)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])

  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

torch.save(model.state_dict(), f'enigma_{n_param:.0f}m.pth')
```

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
For more info, follow [CONTRIBUTE.md](https://github.com/shivendrra/enigma-1.5b/blob/main/CONTRIBUTING.md)
## License
MIT