"""Model for an imitation learning agent."""
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal


class ImitationAgent(nn.Module):
    """Pytorch Module for imitation. Output is a Multivariable Gaussian."""

    def __init__(self, n_states, n_actions, cfg, hidden_layers=[256, 256]):
        """Initialize."""
        super(ImitationAgent, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        self.n_stack = cfg['n_stack']
        self.pos_scaling = cfg['pos_scaling']
        self.heading_scaling = cfg['heading_scaling']
        self.parameters_list = nn.ParameterList([nn.Parameter(torch.tensor([self.pos_scaling, self.pos_scaling, self.heading_scaling]), requires_grad=False)])
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
                nn.Linear(self.hidden_layers[-1], self.n_actions),
                nn.Tanh()
            )

    def dist(self, x):
        """Construct a distirbution from tensor x."""
        x_out = self.nn(x)
        std = torch.ones_like(x_out) * 0.1
        m = MultivariateNormal(x_out[:, 0:self.n_actions] * self.parameters_list[0],
                               torch.diag_embed(std))
        return m

    def forward(self, x, deterministic=False):
        """Generate an output from tensor x."""
        m = self.dist(x)
        if deterministic:
            tanh_output = m.mean
        else:
            tanh_output = m.sample()
        return tanh_output
        
