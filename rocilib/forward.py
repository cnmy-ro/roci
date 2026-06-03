from abc import ABC
import numpy 
import torch

from inverse import *


class ForwardModel(ABC):
    
    def __init__(self):
        ...
    
    def __call__(self, latent):
        ...




# -----
# Forward models

class LinearModel(ForwardModel):
    
    def __init__(self):
        super().__init__()

    def __call__(self, latent):
        ...
        
    def adj(self, obs):
        ...

    def inv(self, obs):
        ...

    def pinv(self, obs):
        ...


class SENSEModel(LinearModel):

    def __init__(self):
        super().__init__()

    def __call__(self, latent):
        ...
        
    def adj(self, obs):
        ...

    def inv(self, obs):
        raise ValueError("Not invertible.")

    def pinv(self, obs):
        ...



# -----
# Noise models

class AdditiveNoiseModel(ForwardModel):
    
    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        ...


class GaussianNoiseModel(AdditiveNoiseModel):

    def __init__(self, stddev):        
        super().__init__()
        self.stddev = stddev

    def __call__(self, obs):
        noise = self.stddev * torch.randn_like(obs)
        return obs + noise
