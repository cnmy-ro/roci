
import torch
import torch.nn as nn



class ComplexMLPFullLength(nn.Module):
    """ Complex-valued MLP implementation based on Virtue et al. "Better than Real" (IEEE Conf. Image Processing 2017). 
    8-layer perceptron. Accepts complex-valued inputs of full length (500 time points).

    Implements phase compensation step described in Cline et al. (2017) and Golbabaee et al. (2019).
    """

    def __init__(self, device='cuda'):
        
        super().__init__()

        self.device = device

        self.blocks = []
        self.blocks.append(ComplexLinearBlock(500, 256, activation=True))
        self.blocks.append(ComplexLinearBlock(256, 256, activation=True))
        self.blocks.append(ComplexLinearBlock(256, 128, activation=True))
        self.blocks.append(ComplexLinearBlock(128, 128, activation=True))
        self.blocks.append(ComplexLinearBlock(128, 64,  activation=True))
        self.blocks.append(ComplexLinearBlock(64, 64,   activation=True))
        self.blocks.append(ComplexLinearBlock(64, 32,   activation=True))
        self.blocks.append(ComplexLinearBlock(32, 2,    activation=False))

        self.model = nn.Sequential(*self.blocks).to(self.device)       

    def forward(self, x):

        # Phase corection layer
        phase_estims = torch.mean(x.angle(), dim=1, keepdims=True)    # Estimated phase = Average phase along time axis. In radians.
        correction_factors = torch.exp(-1j * phase_estims)
        x = x * correction_factors
        
        # Forward pass
        out = self.model(x)
        out = out.abs()

        return out


class ComplexMLPCompressed(nn.Module):
    """ Complex-valued MLP implementation based on Virtue et al. (2017). 
    6-layer perceptron. Small-sized version of `ComplexMLPFullLength`.

    Implements phase compensation step described in Cline et al. (2017) and Golbabaee et al. (2019).
    """

    def __init__(self, num_compression_coeffs, device='cuda'):
        
        super().__init__()

        self.device = device

        self.blocks = []
        self.blocks.append(ComplexLinearBlock(num_compression_coeffs,  256, activation=True))
        self.blocks.append(ComplexLinearBlock(256, 128, activation=True))
        self.blocks.append(ComplexLinearBlock(128, 64,  activation=True))
        self.blocks.append(ComplexLinearBlock(64,  32,  activation=True))
        self.blocks.append(ComplexLinearBlock(32,  16,  activation=True))
        self.blocks.append(ComplexLinearBlock(16,  2,   activation=False))

        self.model = nn.Sequential(*self.blocks).to(self.device)       

    def forward(self, x):
    
        # Phase correction layer
        phase_estims = x[:, 0:1].angle()  # Estimated phase = phase of 1st component. In radians.        
        correction_factors = torch.exp(-1j * phase_estims)
        x = x * correction_factors

        # Forward pass
        out = self.model(x)
        out = out.abs()
        
        return out


class ComplexLinearBlock(nn.Module):
    
    def __init__(self, in_features, out_features, activation=True):
        
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_features, out_features, dtype=torch.cfloat))

        if activation:
            layers.append(ComplexCardioid())

        self.linear_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_block(x)


class ComplexCardioid(nn.Module):
    """ Implementation of the Cardiod activation function for complex-valued nets. 
    Introduced in Virtue et al. "Better than Real" (IEEE Conf. Image Processing 2017). 
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        phase = x.angle()
        attenuation_factor = 0.5 * (1 + torch.cos(phase))
        return attenuation_factor * x