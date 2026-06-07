from abc import ABC
import numpy 
import torch

from observation import *
from inverse import *
from utils import *



class ForwardModel(ABC):
    
    def __init__(self):
        ...
    
    def __call__(self, latent):
        ...



# -----
# Linear models

class LinearModel(ForwardModel):
    
    def __init__(self):
        super().__init__()

    def __call__(self, latent):
        return self.forward(latent)
    
    def forward(self, latent):
        ...
        
    def hermitian(self, obs: Observation):
        ...

    def inv(self, obs: Observation):
        ...

    def pinv(self, obs: Observation):
        ...


class SENSEModel(LinearModel):

    def __init__(self, csm, mask):
        self.csm = csm
        self.mask = mask
        super().__init__()

    def forward(self, image):
        return fft2c(image * self.csm) * self.mask
        
    def hermitian(self, kspace: Observation):
        return torch.sum(ifft2c(kspace * self.mask) * self.csm.conj(), dim=1, keepdim=True)

    def inv(self, kspace: Observation):
        raise ValueError("Not invertible.")

    def pinv(self, kspace: Observation, thresh=1e-10):
        
        """ Compute pseudo-inverse using conjugate gradients """

        def _EHE(x): return self.hermitian(self.forward(x))
        def _conjdot(a,b): return torch.sum(a.conj() * b).real

        a = self.hermitian(kspace)
        b = torch.zeros_like(a)
        r, p = a.clone(), a.clone()
        rdotr = _conjdot(r, r)
        i = 0
        while rdotr > thresh:
            q = _EHE(p)
            b += (rdotr / _conjdot(p, q)) * p
            r -= (rdotr / _conjdot(p, q)) * q
            rdotr_new = _conjdot(r, r)
            p = r + (rdotr_new / rdotr) * p        
            rdotr = rdotr_new
            i += 1
        image = b

        return image



# -----
# Noise models

class AdditiveGaussianNoiseModel(ForwardModel):
    
    def __init__(self, stddev):
        self.stddev = stddev
        super().__init__()

    def __call__(self, latent):
        noise = self.stddev * torch.randn_like(latent)
        return latent + noise
