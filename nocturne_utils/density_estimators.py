"""
Density modules for estimating density of items in the replay buffer (e.g., states / achieved goals).
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.special import entr
import wandb


class RawKernelDensity(object):
  """
  A KDE-based density model for raw items in the replay buffer (e.g., states/goals).
  """
  def __init__(self, optimize_every=10, num_optim_samples=10000, kernel='gaussian', bandwidth=0.1, normalize=True, 
    log_entropy=False, log_figure=False, tag='', buffer_size=1000000,
    quartile_cutoff=0.0, wandb=False, wandb_id=1):
    """[summary]

    Args:
        optimize_every (int, optional): [description]. Defaults to 10.
        num_optim_samples (int, optional): [description]. Number of samples used to optimize the kde.
        kernel (str, optional): [description]. Defaults to 'gaussian'.
        bandwidth (float, optional): [description]. Defaults to 0.1.
        normalize (bool, optional): [description]. Defaults to True.
        log_entropy (bool, optional): [description]. Defaults to False.
        log_figure (bool, optional): [description]. Defaults to False.
        tag (str, optional): [description]. Defaults to ''.
        buffer_size (int, optional): [description]. Defaults to 1000000.
        quartile_cutoff (float, optional): [description]. Defaults to 0.05.
        wandb (bool, optional): [description]. Defaults to False.
        wandb_id (int, optional): used to uniquely index each wandb graph since there are multiple agents.
    """

    self.step = 0
    self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    self.optimize_every = optimize_every
    self.num_optim_samples = num_optim_samples
    self.kernel = kernel
    self.bandwidth = bandwidth
    self.normalize = normalize
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.
    self.fitted_kde = None
    self.ready = False
    self.log_entropy = log_entropy
    self.log_figure = log_figure
    # used to define how low a sample can score and be a valid goal. For example, if the quartile cutoff is 10%, we will sort the 
    # array and return the top element in the 10% of lowest scores.
    # just set it to zero if you want to return the lowest element
    self.quartile_cutoff = quartile_cutoff
    # TODO(eugenevinitsky) make more general
    self.buffer_size = buffer_size
    self.buffer = np.zeros((buffer_size, 2))
    self.pointer = 0
    self.max_pointer_value = 0
    self.wandb = wandb
    self.wandb_id = wandb_id

  def add_sample(self, sample):
      self.buffer[self.pointer] = sample
      self.pointer += 1
      self.pointer %= self.buffer_size
      if self.max_pointer_value < self.buffer_size:
          self.max_pointer_value += 1

  def draw_samples(self, num_samples):
      sample_idxs = np.random.randint(low=0, high=self.max_pointer_value, size=num_samples)
      return self.buffer[sample_idxs]

  def draw_min_sample(self, num_samples):
      samples = self.draw_samples(num_samples)
      scaled_samples = (samples - self.kde_sample_mean) / self.kde_sample_std
      scored_samples = self.evaluate_log_density(scaled_samples)
      valid_value = int(num_samples * self.quartile_cutoff)
      idx = np.argpartition(scored_samples, valid_value)
      valid_idx = np.random.randint(low=0, high=valid_value + 1)
    #   print(f'lowest score was {np.min(scored_samples)}')
      return samples[idx[valid_idx]]

  def _optimize(self, force=False):
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and self.max_pointer_value > self.num_optim_samples):
      self.ready = True
      kde_samples = self.draw_samples(self.num_optim_samples)
      #og_kde_samples = kde_samples

      if self.normalize:
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

      #if self.item == 'ag' and hasattr(self, 'ag_interest') and self.ag_interest.ready:
      #  ag_weights = self.ag_interest.evaluate_disinterest(og_kde_samples)
      #  self.fitted_kde = self.kde.fit(kde_samples, sample_weight=ag_weights.flatten())
      #else:
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now also log the entropy
      if self.log_entropy and self.step % 250 == 0:
        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = 1000
        s = self.fitted_kde.sample(num_samples)
        entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
        # TODO(eugenevinitsky) add to wandb
        if hasattr(self, 'logger'):
            self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)
        else:
            if self.wandb:
              wandb.log({'Explore/{}_entropy_{}'.format(self.module_name, self.wandb_id): entropy})
            print(f'current entropy is {entropy}')

  def plot_density_info(self, num_samples, color, rescaling=1):
    s = self.fitted_kde.sample(num_samples)
    scores = self.evaluate_log_density(s)
    valid_value = int(num_samples * self.quartile_cutoff)
    idx = np.argpartition(scores, valid_value)
    valid_idx = np.random.randint(low=0, high=valid_value + 1)
    s = (s * self.kde_sample_std + self.kde_sample_mean) * rescaling
    sns.kdeplot(s[:, 0], s[:, 1], fill=True, color=color, alpha=0.4)
    plt.scatter(s[idx[valid_idx], 0], s[idx[valid_idx], 1], color=color, s=50)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def evaluate_elementwise_entropy(self, samples, beta=0.):
    """ Given an array of samples, compute elementwise function of entropy of the form:
        elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)
    Args:
      samples: 1-D array of size N
      beta: float, offset entropy calculation
    Returns:
      elem_entropy: 1-D array of size N, elementwise entropy with beta offset
    """
    assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
    log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
    px = np.exp(log_px)
    elem_entropy = entr(px + beta)
    return elem_entropy

  def save(self, save_folder):
    self._save_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)

  def load(self, save_folder):
    self._load_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)
