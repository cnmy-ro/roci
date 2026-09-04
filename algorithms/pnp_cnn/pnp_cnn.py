import numpy as np
import torch
import pywt, ptwt
from tqdm import tqdm
import torch.nn.functional as F




@torch.no_grad()
def pnp_cnn_ista(kspace, mask, csm, max_eig, denoiser, num_iters):


    # Recon
    step_size = 1 / max_eig
    image_estim = sense2d_forward_op_hermitian(kspace, mask, csm)

    for _ in tqdm(range(num_iters)):
        
        # Denoising update
        image_estim = torch.cat([image_estim.real, image_estim.imag], dim=0)
        image_estim = denoiser(image_estim.unsqueeze(0))[0]
        image_estim = image_estim[0] + 1j* image_estim[1]
        image_estim = image_estim.unsqueeze(0)

        # DC update
        image_estim = image_estim - step_size * sense2d_forward_op_hermitian( sense2d_forward_op(image_estim, mask, csm) - kspace, mask, csm )
        
    return image_estim


def pad_to_nearest_divisible_size(image, divisor=32, strict=False, pad_mode='reflect'):

    orig_h, orig_w = image.shape[-2], image.shape[-1]
    candidates_h, candidates_w = np.array([divisor * i for i in range(1000)]), np.array([divisor * i for i in range(1000)])    
    if strict: candidates_h, candidates_w = candidates_h[candidates_h > orig_h], candidates_w[candidates_w > orig_w]
    else:      candidates_h, candidates_w = candidates_h[candidates_h >= orig_h], candidates_w[candidates_w >= orig_w]
    new_h, new_w = candidates_h[np.argmin(candidates_h - orig_h)], candidates_w[np.argmin(candidates_w - orig_w)]

    if (new_h-orig_h)%2 == 0: padding_h_before = padding_h_after = (new_h-orig_h)//2
    else:                     padding_h_before, padding_h_after = (new_h-orig_h)//2, (new_h-orig_h)//2 + 1
    if (new_w-orig_w)%2 == 0: padding_w_before = padding_w_after = (new_w-orig_w)//2
    else:                     padding_w_before, padding_w_after = (new_w-orig_w)//2, (new_w-orig_w)//2 + 1

    if isinstance(image, np.ndarray):
        padding = ((padding_h_before, padding_h_after), (padding_w_before, padding_w_after))
        padded_image = np.pad(image, pad_width=((padding_h_before, padding_h_after), (padding_w_before, padding_w_after)), mode=pad_mode)
    elif isinstance(image, torch.Tensor):
        padding = (padding_w_before, padding_w_after, padding_h_before, padding_h_after)
        padded_image = F.pad(image, pad=padding, mode=pad_mode)

    return padded_image

def unpad(image, orig_size):

    orig_h, orig_w = orig_size[-2], orig_size[-1]
    curr_h, curr_w = image.shape[-2], image.shape[-1]

    from_h = (curr_h-orig_h)//2
    to_h = curr_h - (curr_h-orig_h)//2 if (curr_h-orig_h)%2 == 0 else curr_h - (curr_h-orig_h)//2 - 1
    from_w = (curr_w-orig_w)//2
    to_w = curr_w - (curr_w-orig_w)//2 if (curr_w-orig_w)%2 == 0 else curr_w - (curr_w-orig_w)//2 - 1

    unpadded_image = image[..., from_h:to_h, from_w:to_w]
    return unpadded_image


def sense2d_forward_op(image, csm, mask):
    kspace = fft2c(image * csm, axes=[-2,-1]) * mask
    return kspace


def sense2d_forward_op_hermitian(kspace, csm, mask):
    coil_images = ifft2c(kspace * mask, axes=[-2,-1])
    image = torch.sum(coil_images * csm.conj(), dim=1, keepdim=True)
    return image


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