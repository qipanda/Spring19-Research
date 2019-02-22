import numpy as np
import pandas as pd
import pickle

if __name__ == "__main__":
    # Load resources
    with open("../Data/ABC-News/abc-comp-pos-counts.pickle", "rb") as handle:
        token_counts = pickle.load(handle)
    data = pd.read_csv("../Data/ABC-News/abcnews-complete-pos.txt", sep="\t")
    total_tokens = data.shape[0]

    # Compute subsample discard probabilities per token
    token_discard_probs = dict()
    t = 1e-3
    for token, count in token_counts.items():
        f = count/total_tokens
        token_discard_probs[token] = ((f-t)/f) - (t/f)**0.5
        
    # For each row, probablistically choose to discard or keep it
    with open("../Data/ABC-News/abc-comp-pos-subsample.txt", "w") as f:
        # write header
        f.write("date\tc1\tc2\tword\tpos\n")
        for i, row in data.iterrows():
            print(i)
            # If choose not to discard, write to subsample file
            if np.random.rand() >= token_discard_probs[row["word"]]:
                f.write("{}\t{}\t{}\t{}\t{}\n".format(
                    row["date"], row["c1"], row["c2"], row["word"], True))
