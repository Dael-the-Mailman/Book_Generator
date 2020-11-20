import pandas as pd
from sklearn.utils import shuffle

# Randomizes the order of the dataset from gutenberg.json
df = pd.read_json('../nltk_data/gutenberg.json', orient='records', lines=True)
df = shuffle(df)
df.to_json('../nltk_data/shuffledgutenberg.json', orient='records', lines=True)