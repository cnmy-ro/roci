import torch
import torch.nn as nn


class MoDL(nn.Module):

    def __init__(self, num_blocks, in_channels, device='cuda'):

        super().__init__()
        self.num_blocks = num_blocks
        self.device = device
        self.zscore_normalize = True

        # Shared conv block
        from_img = [nn.Conv2d(in_channels, 64, 3, 1, 1, padding_mode='reflect'), nn.ReLU()]
        to_img = [nn.Conv2d(64, 2, 3, 1, 1, padding_mode='reflect')]
        conv_blocks = []
        for _ in range(4): conv_blocks.extend([nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect'), nn.ReLU()])
        net_block = from_img + conv_blocks + to_img
        net_block = nn.Sequential(*net_block).to(self.device)
        self.net_block = net_block

        # DC regul weight
        self.dc_weight = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, kspace, csm, mask):
        image = sense2d_forward_op_hermitian(kspace, csm, mask)
        for _ in range(self.num_blocks):
            image = self._img_denoise(image)
            image = self._img_dc(self.dc_weight, image, kspace, csm, mask)
        return image

    def _img_denoise(self, image):
        image_input = torch.cat([image.real, image.imag], dim=1)
        if self.zscore_normalize:
            mu = image_input.mean(dim=[-2,-1], keepdim=True)
            sigma = image_input.std(dim=[-2,-1], keepdim=True)
            image_input = (image_input - mu) / (sigma + 1e-8)
        image = image_input[:,0:2,:,:] + self.net_block(image_input)
        if self.zscore_normalize:
            mu2 = image.mean(dim=[-2,-1], keepdim=True)
            image = image - mu2
            image = image*sigma + mu
        image = image[:,0:1,:,:] + 1j*image[:,1:2,:,:]
        return image
    
    def _img_dc(self, dc_weight, image, kspace, csm, mask):
        image = solve_with_CG(image, dc_weight, kspace, csm, mask)
        return image


# Utils

def solve_with_CG(image_regul, regul_weight, kspace, csm, mask):
    
    def EHE(x):
        x_zf = sense2d_forward_op_hermitian( sense2d_forward_op(x, csm, mask), csm, mask )
        return x_zf + regul_weight * x
    
    def conjdot(a,b): return torch.sum(a.conj() * b).real

    a = sense2d_forward_op_hermitian( kspace, csm, mask ) + regul_weight * image_regul
    b = torch.zeros_like(a)
    r, p = a.clone(), a.clone()
    rdotr = conjdot(r, r)
    i = 0
    while i < 10 and rdotr > 1e-10:
        q = EHE(p)
        b += (rdotr / conjdot(p, q)) * p
        r -= (rdotr / conjdot(p, q)) * q
        rdotr_new = conjdot(r, r)
        p = r + (rdotr_new / rdotr) * p        
        rdotr = rdotr_new
        i += 1

    return b

def sense2d_forward_op(image, csm, mask):
    kspace = fft2c(image * csm, axes=[-2,-1]) * mask
    return kspace
def sense2d_forward_op_hermitian(kspace, csm, mask):
    coil_images = ifft2c(kspace * mask, axes=[-2,-1])
    image = torch.sum(coil_images * csm.conj(), dim=1, keepdim=True)
    return image

def fft2c(image, axes=(-2,-1), norm='ortho'):
    kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image, dim=axes), dim=axes, norm=norm), dim=axes)
    return kspace
def ifft2c(kspace, axes=(-2,-1), norm='ortho'):
    image = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=axes), dim=axes, norm=norm), dim=axes) 
    return image