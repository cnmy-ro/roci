import numpy as np
import torch
import pywt, ptwt
from tqdm import tqdm



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


def dwt2(image, wavelet=pywt.Wavelet('db4'), wt_level=4):
    wt_coeffs_real = ptwt.wavedec2(image.real, wavelet, level=wt_level, mode='periodic')
    wt_coeffs_imag = ptwt.wavedec2(image.imag, wavelet, level=wt_level, mode='periodic')
    wt_coeffs = [wt_coeffs_real[0] + 1j*wt_coeffs_imag[0]]
    for level in range(1, wt_level + 1):
        wt_coeffs.append([])
        for i in range(3):
            wt_coeffs[level].append(wt_coeffs_real[level][i] + 1j*wt_coeffs_imag[level][i])
    return wt_coeffs


def idwt2(wt_coeffs, wavelet=pywt.Wavelet('db4')):
    wt_coeffs_real, wt_coeffs_imag = [wt_coeffs[0].real], [wt_coeffs[0].imag]
    for level in range(1, len(wt_coeffs)):
        wt_coeffs_real.append([])
        wt_coeffs_imag.append([])
        for i in range(3):
            wt_coeffs_real[level].append(wt_coeffs[level][i].real)
            wt_coeffs_imag[level].append(wt_coeffs[level][i].imag)
        wt_coeffs_real[level] = ptwt.constants.WaveletCoeff2d(wt_coeffs_real[level])
        wt_coeffs_imag[level] = ptwt.constants.WaveletCoeff2d(wt_coeffs_imag[level])
    wt_coeffs_real = ptwt.constants.WaveletCoeff2d(wt_coeffs_real)
    wt_coeffs_imag = ptwt.constants.WaveletCoeff2d(wt_coeffs_imag)
    image_real = ptwt.waverec2(wt_coeffs_real, wavelet)
    image_imag = ptwt.waverec2(wt_coeffs_imag, wavelet)
    image = image_real + 1j*image_imag
    return image[0]  # 1st dim is batch, and is added by waverec2. Remove it.


def flatten_wavelet_repr(wt_coeffs):
    wt_coeffs_flat = [wt_coeffs[0].flatten()]
    for level in range(1, len(wt_coeffs)):
        for i in range(3):
            wt_coeffs_flat.append(wt_coeffs[level][i].flatten())
    wt_coeffs_flat = torch.cat(wt_coeffs_flat, axis=0)
    return wt_coeffs_flat


def l1_norm(tensor):
    return torch.linalg.norm(tensor.flatten(), ord=1)


def l2_norm(tensor):
    return torch.linalg.norm(tensor.flatten(), ord=2)


def soft_threshold_complex(tensor, alpha):
    # Based on: https://stats.stackexchange.com/questions/357339/soft-thresholding-for-the-lasso-with-complex-valued-data
    return torch.exp(1j * torch.angle(tensor)) * torch.maximum(torch.abs(tensor) - alpha, torch.zeros_like(torch.abs(tensor)))


class ForwardOperator:
    """
    Undersampled 2D Fourier operator
    """
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, estim):
        kspace_estim = fft2c(estim) * self.mask
        return kspace_estim
    
    def hermitian(self, kspace_estim):
        return ifft2c(kspace_estim * self.mask)


class Objective:

    def __init__(self, kspace, forward_op, lambda_l1):
        self.kspace = kspace
        self.forward_op = forward_op
        self.lambda_l1 = lambda_l1

    def __call__(self, image_estim):
        dc_value = self.dc(image_estim)
        l1_value = self.l1(image_estim)
        loss = dc_value + self.lambda_l1 * l1_value
        # print(dc_value, l1_value)
        return loss
    
    def set_device(self, device):
        self.kspace.to(device)
        self.forward_op.mask.to(device)

    def dc(self, image_estim):
        kspace_estim = self.forward_op(image_estim)
        kspace_estim = kspace_estim * self.forward_op.mask # Calculate DC over only the measured samples 
        dc_value = l2_norm(kspace_estim * self.forward_op.mask - self.kspace) ** 2
        return dc_value
    
    def l1(self, image_estim):
        wt_coeffs = dwt2(image_estim)
        wt_coeffs = flatten_wavelet_repr(wt_coeffs)
        l1_value = l1_norm(wt_coeffs)
        return l1_value
    
    def grad_dc(self, image_estim):
        return 2 * self.forward_op.hermitian( self.forward_op(image_estim) - self.kspace )
    
    def prox_l1(self, image_estim, alpha):
        wt_coeffs = dwt2(image_estim)
        wt_coeffs[0] = soft_threshold_complex(wt_coeffs[0], alpha)
        for level in range(1, len(wt_coeffs)):
            for i in range(3):
                wt_coeffs[level][i] = soft_threshold_complex(wt_coeffs[level][i], alpha)
        prox = idwt2(wt_coeffs)
        return prox 


class NLCGSolver:
    
    def __init__(self, max_iters=200, grad_tol=1e-4, alpha=0.05, beta=0.6, max_ls_iters=100, device='cpu'):
        
        self.max_iters = max_iters
        self.grad_tol = grad_tol

        # Line search settings
        self.alpha, self.beta = alpha, beta
        self.max_ls_iters = max_ls_iters   

        self.device = device 
    

    def minimize(self, objective):

        objective.set_device(self.device)
        
        # Disable autograd
        torch.set_grad_enabled(False)

        # Initialize 
        estim = self._init_estimate(objective)
        grad, loss = self._grad_objective(objective, estim)
        delta_estim = -grad
     
        loss_curve = [float(loss.clone().detach().cpu())]

        # Optimization loop
        for it in tqdm(range(self.max_iters)):

            if self._stop_criterion_satisfied(grad):
                break

            # Backtracking line-search            
            t = self._line_search(objective, estim, delta_estim, grad, loss)

            # Optimizer step
            estim_next = estim + t * delta_estim
            grad_next, loss_next = self._grad_objective(objective, estim_next)
            gamma = l2_norm(grad_next) ** 2 / l2_norm(grad) ** 2
            delta_estim_next = - grad_next + gamma * delta_estim

            # Update values for next iter
            estim = estim_next.clone()
            delta_estim = delta_estim_next.clone()
            grad = grad_next.clone()
            loss = loss_next.clone()

            # Record   
            loss_curve.append(float(loss.clone().detach().cpu()))

        # Enable autograd back again to avoid any issues later
        torch.set_grad_enabled(True)
        return estim.detach(), tuple(loss_curve)


    def _init_estimate(self, objective):
        kspace = objective.kspace
        estim = ifft2c(kspace)
        estim.requires_grad = True
        return estim
    

    def _grad_objective(self, objective, estim):
        """
        Compute gradient using autograd.
        """
        torch.set_grad_enabled(True)
        estim = estim.detach()
        if not estim.requires_grad:        
            estim.requires_grad = True
        estim.grad = None
        loss = objective(estim)
        loss.backward()
        torch.set_grad_enabled(False)
        return estim.grad, loss
    

    def _stop_criterion_satisfied(self, grad):
        return l2_norm(grad) < self.grad_tol


    def _line_search(self, objective, estim, delta_estim, grad, loss):
        t, ls_iter = 1, 0
        while True:   
            loss_step = objective(estim + t * delta_estim)
            # This stop condition is slightly different from the paper, and is based on Lustig's MATLAB implementation.
            if loss_step <= loss - self.alpha * t * torch.dot(torch.conj(grad.flatten()), delta_estim.flatten()).abs() \
               or ls_iter > self.max_ls_iters:
                break            
            t *= self.beta
            ls_iter += 1        
        return t
    

class ISTASolver:
    """
    ISTA with optional backtracking and acceleration. 
    
    Ref: Beck and Teboulle (SIAM 2009)
    """
    def __init__(self, num_iters=200, lipschitz_estim=None, fast=True):
        self.num_iters = num_iters
        self.lipschitz_estim = lipschitz_estim  # If not given, do backtracking line-search
        self.fast = fast  # Option for FISTA
        self.do_backtracking = True if lipschitz_estim is None else False 

    def minimize(self, objective):
        
        # Initialize
        image_estim_prev = ifft2c(objective.kspace)
        if self.do_backtracking:
            lipschitz_estim_prev = 1
            eta = 1.001    # TODO: verify init value
        if self.fast:
            y = image_estim_prev
            t = 1

        loss = objective(image_estim_prev)
        loss_curve = [loss]

        # Optimization loop
        for it in tqdm(range(self.num_iters)):

            # Backtracking line-search
            if self.do_backtracking:
                if self.fast: i = self._line_search(objective, y, lipschitz_estim_prev, eta)
                else:         i = self._line_search(objective, image_estim_prev, lipschitz_estim_prev, eta)
                lipschitz_estim = eta**i * lipschitz_estim_prev
                step_size = 1 / lipschitz_estim
            else:
                step_size = 1 / self.lipschitz_estim

            # Optimizer step
            if self.fast:
                image_estim = objective.prox_l1(y - step_size * objective.grad_dc(y), objective.lambda_l1 * step_size)
                t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
                y_next = image_estim + ((t - 1) / t_next) * (image_estim - image_estim_prev)
            else:
                image_estim = objective.prox_l1(image_estim_prev - step_size * objective.grad_dc(image_estim_prev), objective.lambda_l1 * step_size)

            # Update values for next iter
            image_estim_prev = image_estim.clone()            
            if self.do_backtracking: lipschitz_estim_prev = lipschitz_estim
            if self.fast:
                y = y_next.clone()
                t = t_next

            # Record
            loss = objective(image_estim)
            loss_curve.append(loss)

        return image_estim, tuple(loss_curve)

    def _line_search(self, objective, y, lipschitz_estim_prev, eta):
        i = 100  # TODO: verify init value
        while True:
            lipschitz_estim_bar = eta**i * lipschitz_estim_prev
            prox_l1_y = objective.prox_l1(y - 1 / lipschitz_estim_bar * objective.grad_dc(y), objective.lambda_l1 / lipschitz_estim_bar)
            F = objective(prox_l1_y)
            Q = objective.dc(y) + torch.sum((prox_l1_y - y) * objective.grad(y)) + lipschitz_estim_bar / 2 * l2_norm(prox_l1_y - y) + objective.l1(prox_l1_y)            
            if F > Q or i == 1:
                break
            i -= 1
        return i