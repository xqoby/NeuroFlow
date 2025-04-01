import torch
import torch.nn as nn
import torch.nn.functional as F


# MLP Class
class MLP(nn.Module):
    def __init__(self, input_size, n_hidden, hidden_size, output_size):
        super().__init__()
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

        # Apply Xavier initialization to the weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


# AffineTransform Class for Real-NVP
class AffineTransform(nn.Module):
    def __init__(self, type, input_size=2, n_hidden=2, hidden_size=256):
        super().__init__()
        self.mask = self.build_mask(type=type)
        self.scale = nn.Parameter(torch.zeros(1) * 0.1, requires_grad=True)  # Initialize to small value
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.mlp = MLP(input_size=input_size, n_hidden=n_hidden, hidden_size=hidden_size, output_size=2)

    def build_mask(self, type):
        assert type in {"left", "right"}
        if type == "left":
            mask = torch.FloatTensor([1.0, 0.0])
        elif type == "right":
            mask = torch.FloatTensor([0.0, 1.0])
        return mask

    def forward(self, x, reverse=False):
        batch_size = x.shape[0]
        mask = self.mask.repeat(batch_size, x.shape[1] // 2)  # Repeat mask to match the input size
        mask = mask.to(x.device)  # Ensure mask is on the same device as input
        x_ = x * mask

        log_s, t = self.mlp(x_).split(1, dim=1)

        # Apply tanh and scale the log_s to prevent large values
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)

        if reverse:
            x = (x - t) * torch.exp(-log_s)
        else:
            x = x * torch.exp(log_s) + t

        return x, log_s


# RealNVP Class
class RealNVP(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.prior = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))
        self.transforms = nn.ModuleList(transforms)

    def flow(self, x):
        z, log_det = x, torch.zeros_like(x[:, 0])
        for op in self.transforms:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det.sum(dim=1)
        return z, log_det

    def invert_flow(self, z):
        for op in reversed(self.transforms):
            z, _ = op.forward(z, reverse=True)
        return z

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return log_det + torch.sum(self.prior.log_prob(z), dim=1)

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, self.transforms[0].mask.numel()])
        return self.invert_flow(z)

    def nll(self, x):
        return -self.log_prob(x).mean()

    def jacobian_clamping(self, x, lambda_plus=2.0, lambda_minus=1.0):
        noise = torch.randn_like(x) * 1e-5  # Add small noise perturbation
        x_noisy = x + noise

        z1, _ = self.flow(x)
        z2, _ = self.flow(x_noisy)

        # Calculate change ratio Q
        change_ratio = torch.norm(z1 - z2, p=2, dim=1) / torch.norm(noise, p=2, dim=1)
        lambda_plus = torch.tensor(lambda_plus, device=x.device)
        lambda_minus = torch.tensor(lambda_minus, device=x.device)

        # Regularization loss
        L_JC = torch.mean(
            F.relu(change_ratio - lambda_plus) ** 2 + F.relu(lambda_minus - change_ratio) ** 2
        )
        return L_JC



