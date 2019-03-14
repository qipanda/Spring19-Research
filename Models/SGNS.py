import gc
import itertools
import logging
import math
import pandas as pd
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

# CUDA variable for GPU usage if it exists
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# set default torch tensor dtype
torch.set_default_dtype(torch.double)

class SGNSModel(torch.nn.Module):
    def __init__(self, embedding_dim: int, c_vocab_len: int, w_vocab_len: int) -> None:
        """
        Skip-gram Negative Sampling model where c are input target embeddings and
        w are context embeddings. This model serves to define the model for pytorch
        """
        super().__init__()
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
    def __init__(self, s_cnt: int, r_cnt: int, w_cnt: int, K: int, 
                 s_mean: float=0.0, s_std: float=1.0, 
                 r_mean: float=0.0, r_std: float=1.0,
                 w_mean: float=0.0, w_std: float=1.0) -> None:
        """
        s, r, w are source, receiver, and word respectivly, cnts are unique vocab
        length of each and K is their embedding dimension
        """
        super().__init__()
        self.s_embeds = torch.nn.Embedding.from_pretrained(
            torch.nn.init.normal_(torch.empty(s_cnt, K, device=DEVICE), s_mean, s_std), freeze=False)
        self.r_embeds = torch.nn.Embedding.from_pretrained(
            torch.nn.init.normal_(torch.empty(r_cnt, K, device=DEVICE), r_mean, r_std), freeze=False)
        self.w_embeds = torch.nn.Embedding.from_pretrained(
            torch.nn.init.normal_(torch.empty(w_cnt, K, device=DEVICE), w_mean, w_std), freeze=False)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Forward pass through SRModel, adds together the s and receiver tensors,
        applying a non-linearity to each element, then dotting it with the word tensor
        
        X is assumned n x 3 where x is the batch size, 1st col is s, 2nd is r, 3rd is w
        """
        n = X.size()[0]
        prod = torch.bmm(
            (self.s_embeds(X[:, 0]) + self.r_embeds(X[:, 1])).view(n, 1, -1),
            (self.w_embeds(X[:, 2])).view(n, -1, 1))

        return torch.sigmoid(prod).view(n)

class SourceReceiverClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, s_cnt: int=10, r_cnt: int=10, w_cnt: int=100, K: int=5, 
                 s_mean: float=0.0, s_std: float=1.0, r_mean: float=0.0, r_std: float=1.0,
                 w_mean: float=0.0, w_std: float=1.0, lr: float=1e-1, momentum: float=0.0, 
                 weight_decay: float=0.0, dampening: float=0.0, nesterov: bool=False,
                 batch_size: int=32, pred_batch_size: int=100000, train_epocs: int=10, 
                 shuffle: bool=True, torch_threads: int=12, BCE_reduction: str="mean",
                 pred_thresh: float=0.5, log_fpath: str=None) -> None:
        """
        SourceReceiver Classifier wrapper for piping with sklearn.
        """
        # Model parameters
        self.s_cnt = s_cnt
        self.r_cnt = r_cnt
        self.w_cnt = w_cnt
        self.K = K
        self.s_mean = s_mean
        self.s_std = s_std
        self.r_mean = r_mean
        self.r_std = r_std
        self.w_mean = w_mean
        self.w_std = w_std

        # SGD optimization parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        # Other training parameters
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.train_epocs = train_epocs
        self.shuffle = shuffle
        self.torch_threads = torch_threads
        self.BCE_reduction = BCE_reduction
        self.loss_fn = torch.nn.BCELoss(reduction=self.BCE_reduction)

        # Prediction parameters
        self.pred_thresh = pred_thresh

        # Logging parameters
        self.log_fpath = log_fpath

    def returnModel(self):
        """
        Return a blank SR model for this classifier
        """
        return SourceReceiverModel(self.s_cnt, self.r_cnt, self.w_cnt, self.K, 
            self.s_mean, self.s_std, self.r_mean, self.r_std, self.w_mean, self.w_std)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a new SR classifer with data X (samples by features) and 
        y (binary targets)
        """
        # Setup logging to file if available
        if self.log_fpath:
            logging.basicConfig(filename=self.log_fpath, level=logging.INFO)
        logging.info("K={}|lr={:.2E}|wd={:.2E}|CUDA:{}".format(
            self.K, self.lr, self.weight_decay, USE_CUDA))

        # Setup storage for losses
        batches_per_epoch = math.ceil(y.shape[0]/self.batch_size)
        losses = np.zeros(self.train_epocs*batches_per_epoch)

        # Convert X and y to tensors
        X = torch.tensor(X, device=DEVICE)
        y = torch.tensor(y, device=DEVICE)

        # Train a new model
        self.model_ = self.returnModel()

        # Initialize the SGD optimizer
        optimizer = torch.optim.SGD(self.model_.parameters(), 
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
            dampening=self.dampening, nesterov=self.nesterov)

        # set idxs to iterate over per epoch
        idxs = torch.arange(y.size()[0])

        for epoch in range(self.train_epocs):
            logging.info("\tepoch:{}".format(epoch))
            # Shuffle idxs inplace if chosen to do so
            if self.shuffle:
                idxs = idxs[torch.randperm(y.size()[0])]

            for i in itertools.count(0, self.batch_size):
                # Check if gone through whole dataset already
                if i >= y.size()[0]:
                    break

                # Get batch for gradient update
                x, y_target = X[idxs[i:i+self.batch_size], :], y[idxs[i:i+self.batch_size]]

                # Before new batch, zero old gradient instance built up in model
                self.model_.zero_grad()

                # Forward pass to get prob of pos
                pos_prob = self.model_(x)

                # Compute loss function
                loss = self.loss_fn(pos_prob, y_target)

                # Back pass then update based on gradient from back pass
                loss.backward()
                optimizer.step()

                # Log stuff
                batch = int(epoch*batches_per_epoch + i/self.batch_size)
                losses[batch] = loss.item()
                logging.info("\t\tBatch={} of {}|Cum-mean-train-log-loss:{:.4f}".format(
                    int(i/self.batch_size), int(batches_per_epoch), losses.sum()/batch))

        # Free memory of train data and unused cached gpu stuff now that training is done
        del X
        del y
        gc.collect()
        torch.cuda.empty_cache()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of predictions based on [self.pred_thresh]
        """
        y_pred = np.empty(0)
        for i in itertools.count(0, self.pred_batch_size):
            if i >= X.shape[0]:
                break

            x = torch.tensor(X[i:i+self.pred_batch_size, :], device=DEVICE)
            y_batch_pred = self.model_(x).cpu().detach().numpy() > self.pred_thresh
            y_pred = np.concatenate((y_pred, y_batch_pred))

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return list of probabilitic predictions
        """
        y_pred = np.empty(0)
        for i in itertools.count(0, self.pred_batch_size):
            if i >= X.shape[0]:
                break

            x = torch.tensor(X[i:i+self.pred_batch_size, :], device=DEVICE)
            y_batch_pred = self.model_(x).cpu().detach().numpy()
            y_pred = np.concatenate((y_pred, y_batch_pred))

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy average
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
