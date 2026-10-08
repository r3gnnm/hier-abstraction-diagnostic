import torch
import torch.nn as nn


class Abstractor(nn.Module):
    def __init__(self, latent1=128, latent2=32, hidden=64):
        super().__init__()
        self.latent2 = latent2
        self.cell = nn.GRUCell(latent1, latent2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.latent2, device=device)

    def forward(self, z1, z2_prev):
        return self.cell(z1, z2_prev)


class InstantAbstractor(nn.Module):
    """z1_t -> z2_t: INSTANTANEOUS abstraction, z2 = MLP(z1), no memory.

    Same interface as Abstractor (takes and ignores z2_prev) so that both
    variants are trained and probed by exactly the same code. This is the
    'instantaneous' condition of the paper's 2x2 design (table 1).
    """
    def __init__(self, latent1=128, latent2=32, hidden=64):
        super().__init__()
        self.latent2 = latent2
        self.net = nn.Sequential(nn.Linear(latent1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, latent2))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.latent2, device=device)

    def forward(self, z1, z2_prev):
        return self.net(z1)


def make_abstractor(kind, latent1=128, latent2=32):
    """kind: 'recurrent' (GRU) | 'instantaneous' (MLP)"""
    if kind == "recurrent":
        return Abstractor(latent1, latent2)
    if kind == "instantaneous":
        return InstantAbstractor(latent1, latent2)
    raise ValueError(kind)


class Level1Predictor(nn.Module):
    def __init__(self, latent1=128, latent2=32, action_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent1 + action_dim + latent2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent1),
        )

    def forward(self, z1, a, z2):
        return z1 + self.net(torch.cat([z1, a, z2], -1))


class Level2Predictor(nn.Module):
    def __init__(self, latent2=32, action_summary=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent2 + action_summary, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent2),
        )

    def forward(self, z2, a_summary):
        return z2 + self.net(torch.cat([z2, a_summary], -1))


class FlatPredictor(nn.Module):
    def __init__(self, latent1=128, latent2=32, action_dim=2, hidden=288):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent1 + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent1),
        )

    def forward(self, z1, a, z2=None):
        return z1 + self.net(torch.cat([z1, a], -1))


def action_summary(actions):
    """
    actions: (B, k, 2) -> (B, 3)
    """
    s = actions.sum(1)
    dist = actions.norm(dim=-1).sum(1, keepdim=True)
    return torch.cat([s, dist], -1)
