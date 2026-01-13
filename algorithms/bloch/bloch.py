"""
Single-voxel Bloch simulation
"""

import numpy as np
import matplotlib.pyplot as plt


# Constants
PI = np.pi
GAMMA = 42576384.74  # Gyromagnetic ratio of 1H in H20 [Hz/T]
B0 = 0.1           # Main field [T]

# Sim config
REF_FRAME = 'lab'    # 'lab' or 'rot'
DT = 1e-9  # Sim time step [s]; Note: Must be much (~100x) lower than the Nyquist rate 1/(2*GAMMA*Bz) of the precession to avoid sim-related aliasing in viz.


# -----
# Core abstractions

class Voxel:

    def __init__(self, pos=np.array([0,0,0]), t1=4e-3, t2=2e-3, t2star=0, m0=1, m_init=np.array([0,0,1])):
        self.pos = pos # (x,y,z) [m]
        self.t1 = t1   # [s]
        self.t2 = t2   # [s]
        self.t2star = t2star  # [s]  TODO: implement effect
        self.m0 = m0     # [a.u.]
        self.m = m_init  # (x,y,z) [nounits]
        self.num_isochromats = 1  # [nounits] TODO: implement effect

    def read_state(self):
        return self.m
    
    def perturb_state_with_b1(self, alpha):
        phi = np.arctan(self.m[1]/(self.m[0]+1e-8))
        Rrf = self._rfrot(alpha, phi)
        self.m = Rrf @ self.m
    
    def perturb_state_with_b0(self, b0_grad):
        # TODO
        pass
    
    def evolve_state(self):
        Bz = B0    # TODO: add off-resonance effects
        phi = 2 * PI * GAMMA * Bz * DT

        RelA, RelB = self._relax(DT)  # Relaxation
        if REF_FRAME == 'lab':
            Rz = self._zrot(phi)      # Free precession
            RelA, RelB = Rz @ RelA, Rz @ RelB
        
        self.m = RelA @ self.m + RelB

    def _relax(self, t):
        e1, e2 = np.exp(-t/self.t1), np.exp(-t/self.t2)
        A = np.array([
            [e2, 0, 0],
            [0, e2, 0],
            [0, 0, e1]
        ])
        B = np.array([0, 0, self.m0*(1-e1)])
        return A, B

    def _zrot(self, phi):
        Rz = np.array([
            [np.cos(phi), -np.sin(phi), 0], 
            [np.sin(phi), np.cos(phi), 0], 
            [0, 0, 1]
        ])
        return Rz
    def _xrot(self, alpha):
        Rx = np.array([
            [1, 0, 0], 
            [0, np.cos(alpha), -np.sin(alpha)], 
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        return Rx
    def _rfrot(self, alpha, phi):
        Rz1 = self._zrot(-phi)
        Rx = self._xrot(alpha)
        Rz2 = self._zrot(phi)
        Rrf = Rz2 @ Rx @ Rz1
        return Rrf


class Sequence:

    def __init__(self):
        self.total_duration = 1e-5  # [s]
        num_steps = round(self.total_duration / DT)
        self.n_vec = np.arange(0, num_steps)  # Discrete time indices [nounits]
    
    def _apply_rf(self, voxel, alpha):
        voxel.perturb_state_with_b1(alpha)
        return voxel
    
    def _apply_gradient(self, voxel, b0_grad):
        ...

    def run(self, voxel):
        ...


# -----
# Sequences

class FIDSequence(Sequence):
    """ Free induction decay """
    def run(self, voxel):        
        voxel = self._apply_rf(voxel, alpha=PI/2)
        signal = np.empty((self.n_vec.shape[0], 3))
        for n in self.n_vec:
            signal[n] = voxel.read_state()
            voxel.evolve_state()
        return signal


class SESequence(Sequence):
    """ Spin echo TODO """

    def __init__(self, tr, te, num_reps):
        self.tr = tr
        self.te = te
        self.num_reps = num_reps
        num_steps_per_rep = round(tr / DT)
        self.n_vec = np.arange(0, num_steps_per_rep)  # Discrete time indices [nounits]

    def run(self, voxel):        
        signal = np.empty((self.num_reps, 3))
        for r in range(self.num_reps):
            voxel = self._apply_rf(voxel, alpha=PI/2)
            for n in self.n_vec:
                i = self.num_reps * r + n                
                t = i * DT
                if t == self.te:
                    signal[r] = voxel.read_state()
                voxel.evolve_state()
        return signal



# -----
# Run
if __name__ == '__main__':

    # voxel = Voxel()
    # seq = FIDSequence()
    
    voxel = Voxel(t1=1e-5, t2=5e-6)
    seq = SESequence(tr=2e-5, te=1e-5, num_reps=10)
    
    signal = seq.run(voxel)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(signal[:, 0], label='Mx')
    ax.plot(signal[:, 1], label='My')
    ax.plot(signal[:, 2], label='Mz')
    ax.plot(np.linalg.norm(signal, axis=1), label='|M|')
    ax.legend()
    fig.tight_layout()
    plt.show()