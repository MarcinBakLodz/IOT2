import torch
from torchvision import transforms


class TemporalMasking:
    def __init__(self, max_len=10):
        self.max_len = max_len

    def __call__(self, x):
        seq_len = x.shape[0]
        mask_len = torch.randint(1, self.max_len + 1, (1,)).item()
        start = torch.randint(0, seq_len - mask_len, (1,)).item()
        x[start:start+mask_len, :] = 0
        return x
    
class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std
    
    def __call__(self, x):
        return x + torch.randn_like(x) * self.std
    
class ChannelMasking:
    def __init__(self, max_channels=2):
        self.max_channels = max_channels

    def __call__(self, x):
        num_channels = x.shape[1]
        k = torch.randint(1, min(self.max_channels + 1, num_channels), (1,)).item()
        channels_to_mask = torch.randperm(num_channels)[:k]
        x[:, channels_to_mask] = 0
        return x


class NormalizePerChannel:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, x):
        return (x - self.mean) / self.std