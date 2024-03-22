import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class PerCharTokenizer:
  """
  Args:
    - chars (list): all bases along with special tokens represented as characters
    - vocab_size (int): size of vocabulary

  Working:
    - vocab contains all the bases and ['P', 'M', 'U'] as padding, mask and unknown token
    - encode(): iterates over each character a time and the looks up for the position in vocab
      and returns it's position as integer
    - decode(): takes input of a list of integers and returns the specific item from vocab
  """
  def __init__(self):
    super().__init__()
    self.chars = ['\n', 'A', 'T', 'G', 'C', 'P', 'M', 'U', ' ']
    self.vocab_size = len(self.chars)
    self.string_to_index = {ch: i for i, ch in enumerate(self.chars)}
    self.index_to_string = {i: ch for i, ch in enumerate(self.chars)}

  def encode(self, string):
    encoded = []
    for char in string:
      if char in self.string_to_index:
        encoded.append(self.string_to_index[char])
      else:
        special_index = len(self.string_to_index)
        self.string_to_index[char] = special_index
        self.index_to_string[special_index] = char
        encoded.append(special_index)
    return encoded
  
  def decode(self, integer):
    decoded = []
    for i in integer:
      if i in self.index_to_string:
        decoded.append(self.index_to_string[i])
      else:
        continue
    return ''.join(decoded)