import pickle
import pandas as pd
from collections import Counter

if __name__ == "__main__":
    # Get abc complete pos data
    data = pd.read_csv("../Data/ABC-News/abcnews-complete-pos.txt", sep="\t")
    token_counts = Counter()

    for i, row in data.iterrows():
        print(i)
        token_counts[row["word"]] += 1

    # Make vocab for PyTorch
    vocab = dict()
    for i, (key, count) in enumerate(token_counts.items()):
        print(i)
        vocab[key] = i
        
    with open("../Data/ABC-News/abc-comp-pos-counts.pickle", "wb") as handle:
        pickle.dump(token_counts, handle)
    with open("../Data/ABC-News/abc-comp-pos-vocab.pickle", "wb") as handle:
        pickle.dump(vocab, handle)
