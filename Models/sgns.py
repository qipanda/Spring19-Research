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
        c = torch.tensor([self.c_to_idx[c]])
        w = torch.tensor([self.w_to_idx[w]])
        prod = torch.dot(self.c_embeds(c).view(-1), self.w_embeds(w).view(-1))
        prob = torch.sigmoid(prod)
        return prob

if __name__ == "__main__":
    # Load pos and neg training data
    data = pd.read_csv("../Data/ABC-News/abcnews-sgns-processed.txt", sep="\t")

    # Get country pair vocab and word vocab
    country_pair_vocab = (data["c1"] + "-" + data["c2"]).unique()
    word_vocab = data["word"].unique()

    # Create mappings from country pairs and context words to unique integers
    c_to_idx = {}
    w_to_idx = {}
    for idx, c in enumerate(country_pair_vocab):
        c_to_idx[c] = idx
    for idx, w in enumerate(word_vocab):
        w_to_idx[w] = idx

    # Initialize the model
    embedding_dim = 5
    model = SGNS(embedding_dim, c_to_idx, w_to_idx)

    # Train the model
    torch.set_num_threads(5)
    epochs = 10
    loss_fn = torch.nn.BCELoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for i, row in data.iterrows():
            # 1.) Before new datum, need zero old gradient instance
            model.zero_grad()

            # 2.) Forward pass to get prob of positive word
            pos_prob = model(row["c1"] + "-" + row["c2"], row["word"])

            # 3.) Compute loss function
            loss = loss_fn(pos_prob, torch.tensor(row["pos"]))

            # 4.) Back pass to update gradient
            loss.backward()
            optimizer.step()

            # 5.) Log the loss
            total_loss += loss.item()

        # Print loss
        print("epoch:{} | loss:{}".format(epoch, total_loss))

    # Save trained model
    torch.save(model, "../Data/ABC-News/sgns-v1.pt")
