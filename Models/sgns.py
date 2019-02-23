from typing import Dict
import numpy as np
import pandas as pd
import torch

class SGNS(torch.nn.Module):
    def __init__(self, embedding_dim: int, c_to_idx: Dict[str, int],
                 w_to_idx: Dict[str, int],):
        super(SGNS, self).__init__()
        # Store embedding dim dictionaries for country and ctxt word embeddings 
        self.embedding_dim = embedding_dim
        self.c_to_idx = c_to_idx
        self.w_to_idx = w_to_idx

        # Initialize embeddings
        self.c_embeds = torch.nn.Embedding(len(c_to_idx), self.embedding_dim)
        self.w_embeds = torch.nn.Embedding(len(w_to_idx), self.embedding_dim)

    def forward(self, c, w):
        prod = torch.dot(self.c_embeds[self.c_to_idx[c]], self.w_embeds[self.w_to_idx[w]])
        return torch.sigmoid(prod)

if __name__ == "__main__":
    # Load pos and neg training data
    data_pos = pd.read_csv("../Data/ABC-News/abc-comp-pos-subsample.txt", sep="\t")
    data_neg = pd.read_csv("../Data/ABC-News/abc-comp-neg-subsample.txt", sep="\t")

    # Get country pair vocab and word vocab
    country_pair_vocab = (data_pos["c1"] + "-" + data_pos["c2"]).unique()
    word_vocab = data_pos["word"].unique()

    # Create mappings from country pairs and context words to unique integers
    c_to_idx = {}
    w_to_idx = {}
    for idx, c in enumerate(country_pair_vocab):
        c_to_idx[c] = idx
    for idx, w in enumerate(word_vocab):
        w_to_idx[w] = idx

    import ipdb; ipdb.set_trace()
    # Initialize the model
    embedding_dim = 5
    model = SGNS(embedding_dim, c_to_idx, w_to_idx)
