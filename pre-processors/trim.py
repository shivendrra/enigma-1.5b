import os
import pandas as pd
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

def extract_row_data(csv_file):
    df = pd.read_csv(csv_file)
    column_data = df.iloc[:, 0]
    return column_data

def process_csv_files(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_path = os.path.join(directory, csv_file)
        row_data = extract_row_data(csv_path)
        output_file = os.path.splitext(csv_file)[0] + '_extracted_data.txt'
        output_path = os.path.join(directory, output_file)
        with open(output_path, 'w') as f:
            for data in row_data:
                f.write(str(data) + '\n')
            f.write('\n')

directory = '../parquet files/output'
process_csv_files(directory)