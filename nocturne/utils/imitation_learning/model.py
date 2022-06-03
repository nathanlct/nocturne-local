"""Model for an imitation learning agent."""
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical


class ImitationAgent(nn.Module):
    """Pytorch Module for imitation. Output is a Multivariable Gaussian."""

    def __init__(self, n_states, cfg, hidden_layers=[256, 256]):
        """Initialize."""
        super(ImitationAgent, self).__init__()

        self.n_states = n_states
        # self.n_actions = n_actions
        self.cfg = cfg
        self.n_stack = cfg['n_stack']
        self.std_dev = cfg['std_dev']
        self.accel_scaling = cfg['accel_scaling']
        self.steer_scaling = cfg['steer_scaling']
        # self.parameters_list = nn.ParameterList([nn.Parameter(torch.tensor([self.accel_scaling, self.steer_scaling]), requires_grad=False),
        #                                         nn.Parameter(torch.tensor(self.std_dev), requires_grad=False)])
        self.hidden_layers = hidden_layers
        self._build_model()

    def _build_model(self):
        """Build agent MLP that outputs an action mean and variance from a state input."""
        if self.hidden_layers is None or len(self.hidden_layers) == 0:
            self.nn = nn.Linear(self.n_states, self.n_actions)
        else:
            self.nn = nn.Sequential(
                nn.Linear(self.n_states, self.hidden_layers[0]),
                nn.Tanh(),
                *[
                    nn.Sequential(
                        nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                        nn.Tanh(),
                    )
                    for i in range(len(self.hidden_layers) - 1)
                ],
                # nn.Linear(self.hidden_layers[-1], self.n_actions),
                nn.Tanh()
            )
            self.accel_head = nn.Sequential(nn.Linear(self.hidden_layers[-1], self.cfg['accel_discretization']))
            self.steer_head = nn.Sequential(nn.Linear(self.hidden_layers[-1], self.cfg['steer_discretization']))

    def dist(self, x):
        """Construct a distribution from tensor x."""
        x_out = self.nn(x)
        # std = torch.ones_like(x_out) * self.parameters_list[1]
        # m = MultivariateNormal(x_out[:, 0:self.n_actions] * self.parameters_list[0],
        #                        torch.diag_embed(std))
        accel_dist = Categorical(logits=self.accel_head(x_out))
        steer_dist = Categorical(logits=self.steer_head(x_out))
        return accel_dist, steer_dist

    def forward(self, x, deterministic=False):
        """Generate an output from tensor x."""
        accel_dist, steer_dist = self.dist(x)
        if deterministic:
            return (accel_dist.logits.argmax(), steer_dist.logits.argmax())
        else:
            return (accel_dist.sample(), steer_dist.sample())
