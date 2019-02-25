import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from SGNSModel import SGNSModel

class SGNSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_dim: int=5, c_vocab_len: int=100, 
                 w_vocab_len: int=100, lr: float=1e-3, train_epocs: int=10,
                 torch_threads: int=5, BCE_reduction: str="mean"):
        self.embedding_dim = embedding_dim
        self.c_vocab_len = c_vocab_len
        self.w_vocab_len = w_vocab_len
        self.lr = lr
        self.train_epocs = train_epocs
        self.torch_threads = torch_threads
        self.BCE_reduction = BCE_reduction

        self.loss_fn = torch.nn.BCELoss(reduction=self.BCE_reduction)
        self.model = SGNSModel(self.embedding_dim, self.c_vocab_len, self.w_vocab_len)

    def get_params(self, deep=True):
        return {
            "embedding_dim": self.embedding_dim,
            "c_vocab_len": self.c_vocab_len,
            "w_vocab_len": self.w_vocab_len,
            "lr": self.lr,
            "train_epocs": self.train_epocs,
            "torch_threads": self.torch_threads,
            "BCE_reduction": self.BCE_reduction,
        }
        
    def set_params(self, **params):
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key in valid_params:
                valid_params[key] = value 

        # reset model with new params
        self.__init__(**valid_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        INPUTS:
            X - N samples by m feature length
            y - N Binary target labels

        Train the sgns classifer
        """
        torch.set_num_threads(self.torch_threads)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for epoch in range(self.train_epocs):
            for x, y_target in zip(X, y):
                # 1.) Before new datum, zero old gradient instance built up in model
                self.model.zero_grad()

                # 2.) Forward pass to get prob of pos
                pos_prob = self.model(torch.tensor([x[0]]), torch.tensor([x[1]]))

                # 3.) Compute loss function
                loss = self.loss_fn(pos_prob, torch.tensor([y_target], dtype=torch.float))

                # 4.) Back pass then update based on gradient from back pass
                loss.backward()
                optimizer.step()
            
        return self

    def predict(self, X):
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return mean BCE loss based on X and y
        """
        predictions = self.model(torch.tensor(X[:, 0]), torch.tensor(X[:, 1]))
        loss = self.loss_fn(predictions, torch.tensor(y, dtype=torch.float))
        return loss.item() 

