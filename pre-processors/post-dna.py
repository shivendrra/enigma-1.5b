"""
  -> takes input as a input_file/directory and output_file/directory
  -> reads all the dna data present in the file and then writes it in output file
    (without any numbers or any other texts other than 'A', 'T', 'C', 'G')
"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

input_file = 'output/train4.txt'
output_file = 'new_dna.txt'

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
  for line in f_in:
    parts = line.strip().rsplit('\t', 1)
    if len(parts) == 2:
      text = parts[0]
      f_out.write(text + '\n')
    else:
      f_out.write(line)

print(f"Processed data saved to '{output_file}'")