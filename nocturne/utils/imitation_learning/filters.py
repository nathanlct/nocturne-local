import torch
from torch import nn


class MeanStdFilter(nn.Module):
    # adapted from https://www.johndcook.com/blog/standard_deviation/
    def __init__(self, input_shape, eps=1e-05):
        super().__init__()
        self.input_shape = input_shape
        self.eps = eps
        self.track_running_states = True
        self.counter = 0
        self._M = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self._S = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self._n = 0

    def train(self, mode):
        self.track_running_states = True

    def eval(self):
        self.track_running_states = False

    def forward(self, x):
        if self.track_running_states:
            for i in range(x.shape[0]):
                self.push(x[i])
        x = x - self.mean
        x = x / (self.std + self.eps)
        return x

    def push(self, x):
        # Unvectorized update of the running statistics.
        if x.shape != self._M.shape:
            raise ValueError(
                "Unexpected input shape {}, expected {}, value = {}".format(
                    x.shape, self._M.shape, x))
        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            self._M[...] += delta / self._n
            self._S[...] += delta * delta * n1 / self._n

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else torch.square(
            self._M)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape