"""
CG-SENSE

TODO:
- Implement noise decorrelation
- Extend support for non-Cartesian sampling -- do density correction and gridding
"""

import numpy as np
import torch


# ---
# FFTs

def fft2c(image):
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(image, dim=(-2, -1))
    kspace = torch.fft.fftshift(kspace, dim=(-2, -1))
    return kspace

def ifft2c(kspace):
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, dim=(-2, -1))
    image = torch.fft.fftshift(image, dim=(-2, -1))
    return image


# ---
# Linear operators

class LinOp:

    def __init__(self):
        ...

    def __call__(self, x):
        ...

    def H(self, x):
        ...

class SENSE(LinOp):

    def __init__(self, csm, mask):
        self.csm = csm
        self.mask = mask

    def __call__(self, image):
        coil_images = self.csm * image
        kspace = fft2c(coil_images)
        kspace = self.mask * kspace
        return kspace

    def H(self, kspace):
        kspace = self.mask * kspace
        coil_images = ifft2c(kspace)
        image = torch.sum(torch.conj(self.csm) * coil_images, dim=0)
        return image

class IntensityCorrection(LinOp):

    def __init__(self, csm):        
        self.weight_map = 1. / (torch.sqrt(torch.sum(csm.abs()**2, dim=0) + 1e-12))

    def __call__(self, image):
        return self.weight_map * image

    def H(self, image):
        return self.weight_map * image

class DensityCorrection(LinOp):
    # TODO
    def __init__(self):
        pass

    def __call__(self, kspace):
        return kspace

    def H(self, kspace):
        return kspace


# ---
# Utils

def conjdot(a, b):
    return torch.abs(torch.sum(torch.conj(a) * b))


# ---
# CG-SENSE algorithm

def cg_sense(kspace, csm, mask, eps=1e-6):
    
    # Linear operators
    E = SENSE(csm, mask)
    I = IntensityCorrection(csm)
    D = DensityCorrection()

    # Init state
    a = I(E.H(D(kspace)))  # (H, W)
    b = torch.zeros_like(a)  # (H, W)
    p = a.clone()          # (H, W)
    r = a.clone()          # (H, W)

    # Precompute
    rdotr_prev = conjdot(r, r)

    # Loop
    while True:

        delta = rdotr_prev / conjdot(a, a)
        if delta < eps:
            break
        
        q = I(E.H(D(E(I(p)))))
        b += (rdotr_prev / conjdot(p, q)) * p
        r -= (rdotr_prev / conjdot(p, q)) * q
        rdotr = conjdot(r, r)
        p = r + (rdotr / rdotr_prev) * p
        
        rdotr_prev = rdotr

    v = I(b)
    return v
