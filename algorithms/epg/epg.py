from abc import ABC

import numpy as np
import torch



# ---
# Constants

EPS = np.finfo(float).eps



# ---
# Utils

def rounder(val):
    rval = 0
    real = int(round(val.real * 100)) / 100
    imag = int(round(val.imag * 100)) / 100
    if torch.abs(real) > 5*EPS: rval += real
    if torch.abs(imag) > 5*EPS: rval += 1j*imag
    return rval



# ---
# Operators

@torch.no_grad()
def rf_rotate(omega, phi, alpha):
    """
    Mixes coefficients between Fn, F-n and Zn.
    """
    R = torch.tensor([
            [(np.cos(alpha/2))**2,                  np.exp(2*1j*phi)*(np.sin(alpha/2)**2), -1j*np.exp(1j*phi)*np.sin(alpha) ],
            [ np.exp(-2*1j*phi)*np.sin(alpha/2)**2, np.cos(alpha/2)**2,                     1j*np.exp(-1j*phi)*np.sin(alpha)],
            [-1j*0.5*np.exp(-1j*phi)*np.sin(alpha), 1j*0.5*np.exp(1j*phi)*np.sin(alpha),    np.cos(alpha)                   ]
        ],
        device=omega.device, 
        dtype=torch.cfloat)   # Shape: (3,3)
    num_signals, num_fstates = omega.shape[0], omega.shape[2]
    omega = torch.permute(omega, dims=(1,0,2)).reshape(3, num_signals*num_fstates)  # Shape: (batch, 3, Fstates) -> (3, batch, Fstates) -> (3, batch x Fstates)
    omega = torch.mm(R, omega)
    omega = torch.permute(omega.reshape(3, num_signals, num_fstates), dims=(1,0,2)) # Shape: (3, batch x Fstates) -> (3, batch, Fstates) -> (batch, 3, Fstates)
    return omega

@torch.no_grad()
def relax(omega, tau, t1_vec, t2_vec):
    """
    T2 decay attenuates Fn coefficients; 
    T1 recovery attenuates Zn coefficients, and enhances Z0.
    """
    e1_vec, e2_vec = torch.exp(-tau/t1_vec), torch.exp(-tau/t2_vec)  # Shape: (batch)
    zeros, ones = torch.zeros_like(t1_vec), torch.ones_like(t1_vec)
    A = torch.stack([  # Shape: (3, 3, batch)
            torch.stack([e2_vec, zeros,  zeros ], dim=0), 
            torch.stack([zeros,  e2_vec, zeros ], dim=0), 
            torch.stack([zeros,  zeros,  e1_vec], dim=0)
        ], dim=0).to(torch.cfloat)
    A = torch.permute(A, dims=(2,0,1))  # Shape: (batch, 3, 3)
    B = torch.stack([  # Shape: (3, 1, batch)
            torch.stack([zeros        ], dim=0), 
            torch.stack([zeros        ], dim=0), 
            torch.stack([ones - e1_vec], dim=0)
        ], dim=0).to(torch.cfloat)
    B = torch.permute(B, dims=(2,0,1))  # Shape: (batch, 3, 1)
    omega = torch.bmm(A, omega)  # Shape: (batch, 3, 3) x (batch, 3, Fstates) -> (batch, 3, Fstates)
    omega[:, :, 0:1] = omega[:, :, 0:1] + B
    return omega

@torch.no_grad()
def grad_shift(omega, dk):
    """ 
    Increases/decreases Fn state number (n).
    """
    dk = round(dk)
    if dk != 0:
        batch_size, num_fstates = omega.shape[0], omega.shape[2]
        f = torch.cat((torch.flip(omega[:, 0, :], dims=[1]), omega[:, 1, 1:]), dim=1)
        z = omega[:, 2, :]
        zeros = torch.zeros((batch_size, np.abs(dk)), device=omega.device)
        if dk > 0:
            # Positive shift (F+ to the right, F- to the left); i.e. Fn+1=Fn
            f = torch.cat((f, zeros), dim=1)
            fp = torch.flip(f[:, 0 : num_fstates + dk], dims=[1])
            fm = torch.cat((f[:, num_fstates + dk - 1 : ], zeros), dim=1)
            fp[:, 0] = torch.conj(fp[:, 0]).clone()
            z = torch.cat((z, zeros), dim=1)
        else:
            # Negative shift (F- to the right, F+ to the left); i.e. Fn-1=Fn
            f = torch.cat((zeros, f), dim=1)
            fp = torch.cat((torch.flip(f[:, 0 : num_fstates], dims=[1]), zeros), dim=1)
            fm = f[:, num_fstates - 1 : ]
            fm[:, 0] = torch.conj(fm[:, 0]).clone()
            z = torch.cat((z, zeros), dim=1)
        omega = torch.stack((fp, fm, z), dim=1)
    return omega



# ---
# Core EPG class

class EPG:

    def __init__(self, seq, device='cpu'):
        
        self.seq = seq
        self.device = device

        self.t1_list = None
        self.t2_list = None
        self.omega = None
        self.omega_f0_history = None
        self.done = False        

    def simulate(self, t1_list, t2_list):

        self.t1_list, self.t2_list = t1_list.to(self.device), t2_list.to(self.device)
        batch_size = t1_list.shape[0]        
        rf = self.seq.rf
        grad = self.seq.grad
        events = self.seq.events
        timing = self.seq.timing
        uniq_times = np.unique(timing)
        rf_index = 0
        grad_index = 0
        num_events = len(events)

        # Initialize state with (F+(0)=0,F-(0)=0,Z(0)=1)
        self.omega = torch.repeat_interleave(torch.tensor([[[0+0j], [0+0j], [1+0j]]], device=self.device, dtype=torch.cfloat), repeats=batch_size, dim=0)  # Shape: (batch, 3, Fstates)
        self.omega_f0_history = torch.empty((batch_size, num_events), device=self.device, dtype=torch.cfloat)
        
        # Begin simulation
        for t in range(num_events):
            event = events[t]
            if event == "rf":
                phi, alpha = rf[rf_index][0], rf[rf_index][1]
                self.omega = rf_rotate(self.omega, phi, alpha)
                rf_index += 1
            elif event == "grad":
                dk = grad[grad_index]
                self.omega = grad_shift(self.omega, dk)
                grad_index += 1
            elif event == "relax":
                q = np.where(uniq_times == timing[t])[0]
                tau = uniq_times[q] - uniq_times[q-1]
                self.omega = relax(self.omega, tau.item(), self.t1_list, self.t2_list)
            self.omega_f0_history[:, t] = self.omega[:, 0, 0].clone()
        self.done = True
        
    def find_echoes(self):

        assert self.done
        batch_size = self.omega.shape[0]
        timing = torch.tensor(self.seq.timing, device=self.device)
        echoes_batch = self.omega_f0_history.clone()  # Shape: (batch, num_events)        

        # Find echoes
        #   If two non-zero F+'s happen at the same timing, only save the second one as the proper echo
        echoes_batch[:, torch.argwhere( timing[1:]-timing[:-1] < 10*EPS )] = 0
        #   Check for non-zero k=0 state
        echoes_batch = [echoes_batch[b, torch.abs(echoes_batch[b, :]) > 5*EPS].clone() for b in range(batch_size)]
        echoes_batch = torch.stack(echoes_batch, dim=0)

        return echoes_batch  # Shape: (batch, num_echoes)
    
    def reset(self):
        self.t1_list = None
        self.t2_list = None
        self.omega = None
        self.omega_f0_history = None
        self.done = False


# ---
# Sequence classes

class Sequence(ABC):
    def __init__(self, rf, grad, events, timing, name=""):
        self.rf = rf
        self.grad = grad
        self.events = events
        self.timing = timing
        self.name = name


class SpinEchoSequence(Sequence):
    """
    Spin echo sequence
    """
    def __init__(self, alpha, te, tr, reps):

        # Seq params
        self.alpha = alpha
        self.te = te
        self.tr = tr
        self.reps = reps

        # Construct timing
        events = ['rf', 'grad', 'relax', 'rf', 'relax'] * reps
        timing = np.array([0, te/2, te/2, te/2, tr])
        for rep in range(1, reps):
            timing = np.append(timing, np.array([0, te/2, te/2, te/2, tr]) + rep * tr)
        grad = [1] * reps
        rf = [(np.pi/2, np.pi/2), (0, alpha)] * reps
        super().__init__(rf, grad, events, timing, "SE")


class TurboSpinEchoSequence(Sequence):
    """ 
    Turbo spin echo sequence
    """
    def __init__(self, alpha, etl, esp):

        # Seq params
        self.alpha = alpha
        self.etl = etl
        self.esp = esp

        # Construct timing
        events = ['rf', 'grad', 'relax']
        events.extend(etl*['rf', 'grad', 'relax', 'grad', 'relax'])
        timing = np.array([0, esp/2, esp/2])
        timing = np.append(timing, esp/2 + np.array(etl * [0, esp/2, esp/2, esp, esp] + np.repeat(esp * np.arange(etl), 5)))
        grad = (2 * etl + 1) * [1]
        rf = [(np.pi/2, np.pi/2)]    # First 90deg excitation pulse
        rf.extend(etl * [(0, alpha)])  # Pulses to generate the echo train
        super().__init__(rf, grad, events, timing, "TSE")
