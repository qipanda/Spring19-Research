import itertools
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

# CUDA variable for GPU usage if it exists
USE_CUDA = torch.cuda.is_available()

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
        # c and w need to be Nx1 so reshape to force it
        c = c.view(-1, 1)
        w = w.view(-1, 1)
        n = c.size()[0] 

        prods = torch.bmm(self.c_embeds(c).view(n, 1, -1), self.w_embeds(w).view(n, -1, 1))
        probs = torch.sigmoid(prods)
        return probs.view(n)

class SGNSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_dim: int=5, c_vocab_len: int=100, 
                 w_vocab_len: int=100, lr: float=1e-3, batch_size: int=1,
                 train_epocs: int=10, shuffle: bool=True, torch_threads: int=5, 
                 BCE_reduction: str="mean", pred_thresh: float=0.5, 
                 log_fpath: str=None) -> None:
        """
        SGNS Classifier wrapper for piping with sklearn.
        """
        self.embedding_dim = embedding_dim
        self.c_vocab_len = c_vocab_len
        self.w_vocab_len = w_vocab_len
        self.lr = lr
        self.batch_size = batch_size
        self.train_epocs = train_epocs
        self.shuffle = shuffle
        self.torch_threads = torch_threads
        self.BCE_reduction = BCE_reduction
        self.loss_fn = torch.nn.BCELoss(reduction=self.BCE_reduction)
        self.pred_thresh = pred_thresh
        self.log_fpath = log_fpath

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a new sgns classifer with data X (samples by features) and 
        y (binary targets)
        """
        # Setup logging to file if available
        if self.log_fpath:
            logging.basicConfig(filename=self.log_fpath, level=logging.INFO)

        # Train a new model
        self.model_ = SGNSModel(self.embedding_dim, self.c_vocab_len, self.w_vocab_len)

        # If GPU available, use them
        if USE_CUDA:
            self.model_ = self.model_.cuda()

        # set max threads and initialize the SGD optimizer
        torch.set_num_threads(self.torch_threads)
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=self.lr)

        # set idxs to iterate over per epoch
        idxs = np.arange(y.size)

        for epoch in range(self.train_epocs):
            # Shuffle idxs inplace if chosen to do so
            if self.shuffle:
                np.random.shuffle(idxs)

            for i in itertools.count(0, self.batch_size):
                # Check if gone through whole dataset already
                if i >= y.size:
                    break

                # Get batch for gradient update
                x, y_target = X[idxs[i:i+self.batch_size], :], y[idxs[i:i+self.batch_size]]

                # Before new batch, zero old gradient instance built up in model
                self.model_.zero_grad()

                # get input and target as torch tensors, use GPU is available
                c, w = torch.tensor([x[:, 0]]), torch.tensor([x[:, 1]])
                y_target_tensor = torch.tensor([y_target], dtype=torch.float).view(-1)

                if USE_CUDA:
                    c, w = c.cuda(), w.cuda()
                    y_target_tensor = y_target_tensor.cuda()

                # Forward pass to get prob of pos
                pos_prob = self.model_(c, w)

                # Compute loss function
                loss = self.loss_fn(pos_prob, y_target_tensor)

                # Back pass then update based on gradient from back pass
                loss.backward()
                optimizer.step()

                # Log stuff
                logging.info("K:{} | lr:{:.2f} | Epoch:{} | Batch:{} | Train-log-loss:{:.4f}".\
                    format(self.embedding_dim, self.lr, epoch, i, loss.item()))
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of predictions based on [self.pred_thresh]
        """
        c, w = torch.tensor([X[:, 0]]), torch.tensor([X[:, 1]])
        if USE_CUDA:
            c, w = c.cuda(), w.cuda()
            y_pred = self.model_(c, w).cpu().detach().numpy() > self.pred_thresh
        else:
            y_pred = self.model_(c, w).detach().numpy() > self.pred_thresh

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of probabilitic predictions
        """
        c, w = torch.tensor([x[X, 0]]), torch.tensor([X[:, 1]])
        if USE_CUDA:
            c, w = c.cuda(), w.cuda()
            y_pred = self.model_(c, w).cpu().detach().numpy()
        else:
            y_pred = self.model_(c, w).detach().numpy()

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy average
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

class SourceReceiverModel(torch.nn.Module):
    def __init__(self, s_cnt: int, r_cnt: int, w_cnt: int, K: int) -> None:
        """
        s, r, w are source, receiver, and word respectivly, cnts are unique vocab
        length of each and K is their embedding dimension
        """
        super(SourceReceiverModel, self).__init__()
        self.s_embeds = torch.nn.Embedding(s_cnt, K)
        self.r_embeds = torch.nn.Embedding(r_cnt, K)
        self.w_embeds = torch.nn.Embedding(w_cnt, K)

    def forward(self, s: torch.tensor, r: torch.tensor, w: torch.tensor) -> torch.tensor:
        """
        Forward pass through SRModel, adds together the s and receiver tensors,
        applying a non-linearity to each element, then dotting it with the word tensor
        """
        # Force nx1 shape, and save the shape
        s, r, w = s.view(-1, 1), r.view(-1, 1), w.view(-1, 1)
        n = s.size()[0] 

        # Add source and receivers, then dot with word vector for all n samples
        sr_embed = self.s_embeds(s) + self.r_embeds(r)
        w_embed = self.w_embeds(w)
        prods = torch.bmm(sr_embed.view(n, 1, -1), w_embed.view(n, -1, 1))
        probs = torch.sigmoid(prods)

        return probs.view(n)

class SourceReceiverClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, s_cnt: int=10, r_cnt: int=10, w_cnt: int=100, K: int=5,
                 lr: float=1e-1, batch_size: int=32, train_epocs: int=10, 
                 shuffle: bool=True, torch_threads: int=5, BCE_reduction: str="mean",
                 pred_thresh: float=0.5, log_fpath: str=None) -> None:
        """
        SourceReceiver Classifier wrapper for piping with sklearn.
        """
        self.s_cnt = s_cnt
        self.r_cnt = r_cnt
        self.w_cnt = w_cnt
        self.K = K
        self.lr = lr
        self.batch_size = batch_size
        self.train_epocs = train_epocs
        self.shuffle = shuffle
        self.torch_threads = torch_threads
        self.BCE_reduction = BCE_reduction
        self.pred_thresh = pred_thresh
        self.log_fpath = log_fpath

        self.loss_fn = torch.nn.BCELoss(reduction=self.BCE_reduction)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a new SR classifer with data X (samples by features) and 
        y (binary targets)
        """
        # Setup logging to file if available
        if self.log_fpath:
            logging.basicConfig(filename=self.log_fpath, level=logging.INFO)

        # Train a new model
        self.model_ = SourceReceiverModel(self.s_cnt, self.r_cnt, self.w_cnt, self.K)

        # If GPU available, use them
        if USE_CUDA:
            self.model_ = self.model_.cuda()

        # set max threads and initialize the SGD optimizer
        torch.set_num_threads(self.torch_threads)
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=self.lr)

        # set idxs to iterate over per epoch
        idxs = np.arange(y.size)

        for epoch in range(self.train_epocs):
            # Shuffle idxs inplace if chosen to do so
            if self.shuffle:
                np.random.shuffle(idxs)

            for i in itertools.count(0, self.batch_size):
                # Check if gone through whole dataset already
                if i >= y.size:
                    break

                # Get batch for gradient update
                x, y_target = X[idxs[i:i+self.batch_size], :], y[idxs[i:i+self.batch_size]]

                # Before new batch, zero old gradient instance built up in model
                self.model_.zero_grad()

                # get input and target as torch tensors, use GPU is available
                s, r, w = torch.tensor([x[:, 0]]), torch.tensor([x[:, 1]]), torch.tensor([x[:, 2]])
                y_target_tensor = torch.tensor([y_target], dtype=torch.float).view(-1)

                if USE_CUDA:
                    s, r, w = s.cuda(), r.cuda(), w.cuda()
                    y_target_tensor = y_target_tensor.cuda()

                # Forward pass to get prob of pos
                pos_prob = self.model_(s, r, w)

                # Compute loss function
                loss = self.loss_fn(pos_prob, y_target_tensor)

                # Back pass then update based on gradient from back pass
                loss.backward()
                optimizer.step()

                # Log stuff
                logging.info("K:{} | lr:{:.2f} | Epoch:{} | Batch:{} | Train-log-loss:{:.4f}".\
                    format(self.K, self.lr, epoch, i/self.batch_size, loss.item()))
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of predictions based on [self.pred_thresh]
        """
        s, r, w = torch.tensor([X[:, 0]]), torch.tensor([X[:, 1]]), torch.tensor([X[:, 2]])
        if USE_CUDA:
            s, r, w = s.cuda(), r.cuda(), w.cuda()
            y_pred = self.model_(s, r, w).cpu().detach().numpy() > self.pred_thresh
        else:
            y_pred = self.model_(s, r, w).detach().numpy() > self.pred_thresh

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of probabilitic predictions
        """
        s, r, w = torch.tensor([X[:, 0]]), torch.tensor([X[:, 1]]), torch.tensor([X[:, 2]])
        if USE_CUDA:
            s, r, w = s.cuda(), r.cuda(), w.cuda()
            y_pred = self.model_(s, r, w).cpu().detach().numpy()
        else:
            y_pred = self.model_(s, r, w).detach().numpy()

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy average
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
