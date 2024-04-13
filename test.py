import regex as re

def _tokenize_sequence(sequence, k_mers, tokenize_special_tokens=False):
    if tokenize_special_tokens:
        tokens = re.findall(rf"<unk>|<pad>|<mask>|.{{1,{k_mers}}}", sequence)
    else:
        tokens = [sequence[i:i+k_mers] for i in range(0, len(sequence), k_mers)]
    return tokens

def tokenize_sequence(seq, k_mers, tokenize_special_tokens=False):
  kmers = []
  for i in range(0, len(seq), k_mers):
    kmers.append(seq[i:i+k_mers])
  return kmers

sequence = """ATGGCCTCGCGC<mask>TGGTGGCGGTGGCGAC<pad>GCGGCTGCTCCTGGAGGCCGGCGGCGCGGAGCTCCG<unk>"""
outputs = tokenize_sequence(sequence, tokenize_special_tokens=True, k_mers=6)
output = _tokenize_sequence(sequence, tokenize_special_tokens=True, k_mers=3)

print(outputs, '\n', output)