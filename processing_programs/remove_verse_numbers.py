import re
import pandas as pd

def verse_present(text):
    return re.sub(r'\d+:\d+', '',text)

df = pd.read_json('../nltk_data/shuffledgutenberg.json', orient='records', lines=True)
df["Input"] = df["Input"].apply(verse_present)
df["Output"] = df["Output"].apply(verse_present)
df.to_json('../nltk_data/finalgutenberg.json', orient='records', lines=True)