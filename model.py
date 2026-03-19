import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    The initial block: 
    - Conv 256 filters, 3x3, stride 1
    - Batch Norm
    - ReLU
    """
    def __init__(self, in_channels, out_channels=256):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """
    A single residual block:
    - Conv 256 (3x3) -> BN -> ReLU
    - Conv 256 (3x3) -> BN
    - Skip Connection (Input + Output)
    - ReLU
    """
    def __init__(self, channels=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # The Skip Connection
        return F.relu(out)

class SquadroNet(nn.Module):
    def __init__(self, num_res_blocks=10, in_channels=5):
        super(SquadroNet, self).__init__()
        
        # --- BODY ---
        # 1. Initial Convolutional Block
        self.conv_block = ConvBlock(in_channels, 256)
        
        # 2. 10 Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_res_blocks)
        ])
        
        # --- POLICY HEAD ---
        # Conv 256 filters (1x1) -> BN -> ReLU -> Linear -> Action Size (5)
        self.policy_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(256)
        # Linear input size: 256 filters * 5x5 board = 6400
        self.policy_fc = nn.Linear(256 * 5 * 5, 5) 
        
        # --- VALUE HEAD ---
        # Conv 1 filter (1x1) -> BN -> ReLU -> Linear(256) -> ReLU -> Linear(1) -> Tanh
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        # Linear input size: 1 filter * 5x5 board = 25
        self.value_fc1 = nn.Linear(1 * 5 * 5, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (Batch, 5, 5, 5) -> 5 input planes
        
        # Pass through Body
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        
        # Pass through Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)
        
        # Pass through Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v