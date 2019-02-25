from typing import Dict
import numpy as np
import pandas as pd
import torch
import pickle

class SGNSModel(torch.nn.Module):
    def __init__(self, embedding_dim: int, c_vocab_len: int, w_vocab_len: int):
        """
        c is input vocab len and w is output vocab len
        """
        super(SGNSModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.c_embeds = torch.nn.Embedding(c_vocab_len, self.embedding_dim)
        self.w_embeds = torch.nn.Embedding(w_vocab_len, self.embedding_dim)

    def forward(self, c: torch.tensor, w: torch.tensor) -> torch.tensor:
        n = c.size()[0] # c and w should have same first time
        prods = torch.bmm(self.c_embeds(c).view(n, 1, -1), self.w_embeds(w).view(n, -1, 1))
        probs = torch.sigmoid(prods)
        return probs.view(n)
