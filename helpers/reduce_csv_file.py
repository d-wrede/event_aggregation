"""This script reads in the CSV file containing the word frequencies and filters it to only include words with a frequency above 400,000. The filtered DataFrame is then saved to a new CSV file. The script also calculates the number of words in the filtered DataFrame and the number of words that have been filtered out."""


import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, "..", "decow_wordfreq_cistem.csv")
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, index_col=["word"])

# Filter the DataFrame to only include rows with a frequency above 600,000
filtered_df = df[df["freq"] > 400000]

# Save the filtered DataFrame to a new CSV file
csv_file_path = os.path.join(current_dir, "..", "filtered_decow_wordfreq_cistem.csv")
filtered_df.to_csv(csv_file_path)
print("saved filtered df")

# Calculate the number of words in the filtered DataFrame and the number of words that have been filtered out
num_words_filtered = len(filtered_df)
num_words_filtered_out = len(df) - num_words_filtered

print(f"Number of words in the filtered DataFrame: {num_words_filtered}")
print(f"Number of words filtered out: {num_words_filtered_out}")
