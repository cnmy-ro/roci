import numpy as np
import torch


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


def sense(coil_images, sens_maps, acceleration):
    """
    SENSE coil combination: Naive implementation
    """
    image_sense = torch.full(coil_images.shape[-2:], torch.nan, dtype=torch.complex64)

    num_cols = coil_images.shape[-1]
    num_rows = coil_images.shape[-2]
    for col in range(num_cols):  # for each col
        
        # If the column is filled, then skip
        if not torch.isnan(torch.sum(image_sense[:, col])):
            continue

        superimp_cols = torch.tensor([col + int(i/acceleration * num_cols) for i in range(acceleration)]) # Size (Np,)

        for row in range(num_rows):
            
            S = sens_maps[:, row, superimp_cols]  # Size (Nc x Np)
            U = torch.linalg.pinv(S) # Size (Np x Nc)
            a = coil_images[:, row, col] # Size (Nc,)
            v = torch.dot(U, a) # Size (Np,)

            image_sense[row, superimp_cols] = v

    return image_sense


def sense_vectorized(coil_images, sens_maps, acceleration):
    """
    SENSE coil combination: Vectorized implementation
    """
    image_sense = torch.full(coil_images.shape[-2:], torch.nan, dtype=torch.complex64)

    num_cols = coil_images.shape[-1]
    for col in range(num_cols):  # for each col
        
        # If the column is filled, then skip
        if not torch.isnan(torch.sum(image_sense[:, col])):
            continue

        superimp_cols = torch.tensor([col + int(i/acceleration * num_cols) for i in range(acceleration)]) # Size Np
        
        # Vectorized operations
        S = sens_maps[:, :, superimp_cols]  # Size (Nc x num_rows x Np)
        S = S.permute(1,0,2) # Size (num_rows x Nc x Np)
        U = torch.linalg.pinv(S) # Size (num_rows x Np x Nc)
        a = coil_images[:, :, col] # Size (Nc x num_rows)
        a = a.unsqueeze(0).permute(2,1,0) # Size (num_rows x Nc x 1)
        v = torch.matmul(U, a) # Size (num_rows x Np x 1)
        image_sense[:, superimp_cols] = v.squeeze()

    return image_sense