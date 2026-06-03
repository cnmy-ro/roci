from abc import ABC
import numpy 
import torch



class DiscreteRepr(ABC):
    """
    Essentially a finite-dimensional vector
    """
    def __init__(self):
        ...
    

class ContinuousRepr(ABC):
    """
    Essentially a scalar field
    """
    def __init__(self):
        ...
    
