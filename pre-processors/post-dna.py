"""
  -> takes input as a input_file/directory and output_file/directory
  -> reads all the dna data present in the file and then writes it in output file
    (without any numbers or any other texts other than 'A', 'T', 'C', 'G')
"""

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

def process_dna_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            dna_data = ''.join(char for char in line.strip() if char in ['A', 'T', 'C', 'G'])
            f_out.write(dna_data + '\n')

def process_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_dna_data(input_file, output_file)
            print(f"Processed file '{input_file}' saved to '{output_file}'")

if __name__ == "__main__":
    input_directory = '../output'
    output_directory = '../processed outputs'

    process_files(input_directory, output_directory)