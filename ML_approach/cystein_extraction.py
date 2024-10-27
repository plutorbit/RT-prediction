import pandas as pd

input_file = pd.read_csv('new_data.csv')
input_file.head()


def count_cystein(sequence):
    return sequence.count('C')

input_file['cystein_count'] = input_file['sequence'].apply(count_cystein)

input_file.to_csv('new_data_cystein.csv', index=False)