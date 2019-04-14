import gc
import itertools
import math
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorboardX import SummaryWriter

# CUDA variable for GPU usage if it exists
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# set default torch tensor dtype
torch.set_default_dtype(torch.double)

class SRCTModel(torch.nn.Module):
    def __init__(self, s_cnt: int, r_cnt: int, p_cnt: int, K_s: int, K_r: int, K_p: int,
                 T: int) -> None:
        """
        Source-Receiver model letting the s/r embedding be concatenated and have
        a timesteps. Each source, receiver, and predicate embedding consequently 
        has their own K. Source and Receiver hidden dims should add to the word 
        hidden dim.
        """
        super().__init__()
        self.s_cnt = s_cnt
        self.r_cnt = r_cnt
        self.p_cnt = p_cnt
        self.T = T

        self.s_embeds = torch.nn.Embedding.from_pretrained(
            torch.nn.init.xavier_uniform_(torch.empty(T*s_cnt, K_s, device=DEVICE)), freeze=False) 
        self.r_embeds = torch.nn.Embedding.from_pretrained(
            torch.nn.init.xavier_uniform_(torch.empty(T*r_cnt, K_r, device=DEVICE)), freeze=False) 
        self.p_embeds = torch.nn.Embedding.from_pretrained(
            torch.nn.init.xavier_uniform_(torch.empty(p_cnt, K_p, device=DEVICE)), freeze=False) 

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Forward pass through SRCT model, concats together s and r tensors from 
        give t and then applies dot product of that concatenation with the p tensor

        X is assumed n x 4 where x is the batch size, 1st col is s, 2nd is r, 3rd is p, 4th is t
        """
        n = X.size()[0]
        s, r, p, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        st, rt = s + t*self.s_cnt, r + t*self.r_cnt

        prod = torch.bmm(
            torch.cat((self.s_embeds(st), self.r_embeds(rt)), dim=1).view(n, 1, -1),
            (self.p_embeds(p)).view(n, -1, 1))

        return torch.sigmoid(prod).view(n)

    def returnRegTerm(self, alpha: float, lam: float):
        """
        Return the L2 loss of all model parameters(embeddings) along with the temporal
        L2 loss between source and receiver embeddings one time step away
        """
        L2_params = torch.norm(self.s_embeds.weight, 2)**2.0 + \
            torch.norm(self.r_embeds.weight, 2)**2.0 + \
            torch.norm(self.p_embeds.weight, 2)**2.0

        L2_time = \
            torch.norm(self.s_embeds.weight[:self.s_cnt*(self.T-1), :] - 
                self.s_embeds.weight[self.s_cnt:, :], 2)**2.0 + \
            torch.norm(self.r_embeds.weight[:self.r_cnt*(self.T-1), :] -
                self.r_embeds.weight[self.r_cnt:, :], 2)**2.0

        return lam*L2_params + alpha*L2_time

class SRCTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, s_cnt: int=10, r_cnt: int=10, p_cnt: int=100, T: int=1,
                 K: int=None, K_s: int=25, K_r: int=25, K_p: int=50,
                 lr: float=1e-1, alpha: float=0.5, lam: float=0.0,
                 batch_size: int=32, pred_batch_size: int=100000, train_epochs: int=1, 
                 shuffle: bool=True, torch_threads: int=12, BCE_reduction: str="mean",
                 pred_thresh: float=0.5, log_fpath: str=None, hist_mod: int=100) -> None:
        """
        SourceReceiver Classifier wrapper for piping with sklearn.
        """
        # Model parameters
        self.s_cnt = s_cnt
        self.r_cnt = r_cnt
        self.p_cnt = p_cnt
        self.T = T
        if K is not None:
            self.K_s = K//2
            self.K_r = K//2
            self.K_p = K
        else:
            self.K_s = K_s
            self.K_r = K_r
            self.K_p = K_p

        # SGD optimization parameters
        self.lr = lr
        self.lam = lam
        self.alpha = alpha

        # Other training parameters
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.train_epochs = train_epochs
        self.shuffle = shuffle
        self.torch_threads = torch_threads
        self.BCE_reduction = BCE_reduction
        self.loss_fn = torch.nn.BCELoss(reduction=self.BCE_reduction)

        # Prediction parameters
        self.pred_thresh = pred_thresh

        # Logging parameters
        self.log_fpath = log_fpath
        self.hist_mod = hist_mod
        self.tensorboard_path = "K{}_lr{:.2E}_lam{:.2E}_alpha{:.2E}_bs{}_epochs{}".format(
            self.K_p, self.lr, self.lam, self.alpha, self.batch_size, self.train_epochs)
        self.writer = SummaryWriter(log_dir=(self.log_fpath + "/" + self.tensorboard_path))

    def returnModel(self):
        """
        Return a blank SRCT model for this classifier
        """
        return SRCTModel(self.s_cnt, self.r_cnt, self.p_cnt, self.K_s, self.K_r, self.K_p, self.T)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a new SRCT classifer with data X (samples by features) and y (binary targets)
        """
        # For logging purposes
        batches_per_epoch = math.ceil(y.shape[0]/self.batch_size)

        # Convert X and y to tensors
        X = torch.tensor(X, device=DEVICE)
        y = torch.tensor(y, device=DEVICE)

        # Train a new model
        self.model_ = self.returnModel()

        # Initialize the SGD optimizer
        optimizer = torch.optim.SGD(self.model_.parameters(), lr=self.lr)

        # set idxs to iterate over per epoch
        idxs = torch.arange(y.size()[0])

        for epoch in range(self.train_epochs):
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
                NLL = self.loss_fn(pos_prob, y_target)
                REG = self.model_.returnRegTerm(self.alpha, self.lam)
                loss = NLL + REG

                # Back pass then update based on gradient from back pass
                loss.backward()
                optimizer.step()

                # Tensoboard Log stuff
                cum_batch_id = int(epoch*batches_per_epoch + i/self.batch_size)
                self.writer.add_scalar("train-loss", NLL.item(), cum_batch_id)
                self.writer.add_scalar("reg-loss", REG.item(), cum_batch_id)
                self.writer.add_scalar("total-loss", loss.item(), cum_batch_id)

                if cum_batch_id % self.hist_mod == 0:
                    self.writer.add_histogram(tag="s_embeds",
                                              values=self.model_.s_embeds.weight.detach(),
                                              global_step=cum_batch_id,)
                    self.writer.add_histogram(tag="r_embeds",
                                              values=self.model_.r_embeds.weight.detach(),
                                              global_step=cum_batch_id,)
                    self.writer.add_histogram(tag="p_embeds",
                                              values=self.model_.p_embeds.weight.detach(),
                                              global_step=cum_batch_id,)

        # # Free memory of train data and unused cached gpu stuff now that training is done
        # del X
        # del y
        # gc.collect()
        # torch.cuda.empty_cache()

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

class SRCTSoftmaxModel(SRCTModel):
    def __init__(self, s_cnt: int, r_cnt: int, p_cnt: int, K_s: int, K_r: int, K_p: int,
                 T: int) -> None:
        """
        Source-Receiver model letting the s/r embedding be concatenated and have
        a timesteps. Each source, receiver, and predicate embedding consequently 
        has their own K. Source and Receiver hidden dims should add to the word 
        hidden dim.
        """
        super().__init__(s_cnt, r_cnt, p_cnt, K_s, K_r, K_p, T)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Forward pass through SRCT model, concats together s and r tensors from 
        give t and then applies dot product of that concatenation with the p embeds
        to get Pr(p|s,r,t) for all p (prob distribution).

        X is assumed n x 3 where x is the batch size, 1st col is s, 2nd is r, 3rd is t

        return batch x p of non-softmaxed probs
        """
        n = X.size()[0]
        s, r, t = X[:, 0], X[:, 1], X[:, 2]
        st, rt = s + t*self.s_cnt, r + t*self.r_cnt

        prod = torch.matmul(
            torch.cat((self.s_embeds(st), self.r_embeds(rt)), dim=1),
            self.p_embeds.weight.transpose(1, 0))

        return self.logsoftmax(prod)

class SRCTSoftmaxClassifier(SRCTClassifier):
    def __init__(self, s_cnt: int=10, r_cnt: int=10, p_cnt: int=100, T: int=1,
                 K: int=None, K_s: int=25, K_r: int=25, K_p: int=50,
                 lr: float=1e-1, alpha: float=0.5, lam: float=0.0,
                 batch_size: int=32, pred_batch_size: int=100000, train_epochs: int=1, 
                 shuffle: bool=True, torch_threads: int=12, BCE_reduction: str="mean",
                 pred_thresh: float=0.5, log_fpath: str=None, hist_mod: int=100) -> None:
        """
        SourceReceiver Classifier wrapper for piping with sklearn.
        """
        super().__init__(s_cnt, r_cnt, p_cnt, T, K, K_s, K_r, K_p, lr, alpha,
                         lam, batch_size, pred_batch_size, train_epochs, shuffle, 
                         torch_threads, BCE_reduction, pred_thresh, log_fpath, hist_mod)
        self.loss_fn = torch.nn.NLLLoss(reduction=self.BCE_reduction)
        self.tensorboard_path = "softmax_K{}_lr{:.2E}_lam{:.2E}_alpha{:.2E}_bs{}_epochs{}".format(
            self.K_p, self.lr, self.lam, self.alpha, self.batch_size, self.train_epochs)
        self.writer = SummaryWriter(log_dir=(self.log_fpath + "/" + self.tensorboard_path))

    def returnModel(self):
        """
        Return a blank SRCTSoftmax model for this classifier
        """
        return SRCTSoftmaxModel(self.s_cnt, self.r_cnt, self.p_cnt, self.K_s,
            self.K_r, self.K_p, self.T)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO predict, predict_proba, and score need to be adjusted
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
