import pandas as pd
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

class SGNSModel(torch.nn.Module):
    def __init__(self, embedding_dim: int, c_vocab_len: int, w_vocab_len: int) -> None:
        """
        Skip-gram Negative Sampling model where c are input target embeddings and
        w are context embeddings. This model serves to define the model for pytorch
        """
        super(SGNSModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.c_embeds = torch.nn.Embedding(c_vocab_len, self.embedding_dim)
        self.w_embeds = torch.nn.Embedding(w_vocab_len, self.embedding_dim)

    def forward(self, c: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Forward pass through the sgns model, essentially the dot product of relevant
        embeddings. First dim of c and w indicate how many samples we are putting 
        through the forward pass and we return the prob(+|w,c)
        """
        n = c.size()[0] # c and w should have same first dim
        prods = torch.bmm(self.c_embeds(c).view(n, 1, -1), self.w_embeds(w).view(n, -1, 1))
        probs = torch.sigmoid(prods)
        return probs.view(n)

class SGNSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_dim: int=5, c_vocab_len: int=100, 
                 w_vocab_len: int=100, lr: float=1e-3, train_epocs: int=10,
                 torch_threads: int=5, BCE_reduction: str="mean", 
                 pred_thresh: float=0.5) -> None:
        """
        SGNS Classifier wrapper for piping with sklearn.
        """
        self.embedding_dim = embedding_dim
        self.c_vocab_len = c_vocab_len
        self.w_vocab_len = w_vocab_len
        self.lr = lr
        self.train_epocs = train_epocs
        self.torch_threads = torch_threads
        self.BCE_reduction = BCE_reduction
        self.loss_fn = torch.nn.BCELoss(reduction=self.BCE_reduction)
        self.pred_thresh = pred_thresh

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a new sgns classifer with data X (samples by features) and 
        y (binary targets)
        """
        # Train a new model
        self.model_ = SGNSModel(self.embedding_dim, self.c_vocab_len, self.w_vocab_len)

        torch.set_num_threads(self.torch_threads)
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=self.lr)

        for epoch in range(self.train_epocs):
            for x, y_target in zip(X, y):
                # 1.) Before new datum, zero old gradient instance built up in model
                self.model_.zero_grad()

                # 2.) Forward pass to get prob of pos
                pos_prob = self.model_(torch.tensor([x[0]]), torch.tensor([x[1]]))

                # 3.) Compute loss function
                loss = self.loss_fn(pos_prob, torch.tensor([y_target], dtype=torch.float))

                # 4.) Back pass then update based on gradient from back pass
                loss.backward()
                optimizer.step()
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of predictions based on [self.pred_thresh]
        """
        return self.model_(torch.tensor(X[:, 0]), torch.tensor(X[:, 1]))\
            .detach().numpy() > self.pred_thresh

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of probabilitic predictions
        """
        return self.model_(torch.tensor(X[:, 0]), torch.tensor(X[:, 1]))\
            .detach().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy average
        """
        predictions = self.model_(torch.tensor(X[:, 0]), torch.tensor(X[:, 1]))
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
