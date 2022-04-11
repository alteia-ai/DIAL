import torch.nn as nn
import torch.nn.functional as F

class ConfidNet(nn.Module):
    def __init__(self):
        super(ConfidNet, self).__init__()
        self.unpool_uncertainty = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        self.uncertainty = nn.Conv2d(32, 64, 3, 1, 1)

        self.uncertainty1 = nn.Conv2d(32, 120, 3, 1, 1)
        self.uncertainty2 = nn.Conv2d(400, 120, 3, 1, 1)
        self.uncertainty3 = nn.Conv2d(120, 64, 3, 1, 1)
        self.uncertainty4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.uncertainty5 = nn.Conv2d(64, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        uncertainty = self.unpool_uncertainty(x)
        uncertainty = F.relu(self.uncertainty1(uncertainty))
        uncertainty = F.relu(self.uncertainty3(uncertainty))
        uncertainty = F.relu(self.uncertainty4(uncertainty))
        uncertainty = self.uncertainty5(uncertainty)
        uncertainty = self.sigmoid(uncertainty)
        return uncertainty