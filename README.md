<p align="center">
  <img src="docs/roci_logo.png"  width="400">
</p>

[![DOI](https://zenodo.org/badge/1132861288.svg)](https://doi.org/10.5281/zenodo.18778617)

Repository of computational imaging (`roci`) is a collection of high-quality and self-contained (re-)implementations of algorithms used in computational imaging. In terms of applications, it currently focuses on forward and inverse problems in computational MRI (reconstruction, quantification, synthesis, physics simulation). In terms of techniques, `roci` includes classical signal processing algorithms as well as state-of-the-art representation learning and generative modeling methods. For educational and research purposes only.


## Why `roci`

Original software implmentations offered by authors of papers vary largely in quality and style, may be incomplete, or implemented in a less popular language or library. This introduces difficulty for students studying them and for reseachers and enginners adapting them for their own use. `roci` bridges this gap by providing software re-implementations that are concise, self-contained, and standardized in style.


## Code Organization

The directory structure is simple: `algorithm = python_file + demo_notebook + readme`.
```
algorithms
    |
    |- algo_1
    |   |- algo_1.py
    |   |- Demo.ipynb
    |   |- README.md
    |
    |- algo_2
    |   |- algo_2.py
    |   |- Demo.ipynb
    |   |- README.md
    ...
```

## Available Algorithms

MRI reconstruction:
- [SENSE parallel-imaging reconstruction](algorithms/sense/)
- [CG-SENSE parallel-imaging reconstruction](algorithms/cg_sense/)
- [Compressed sensing reconstruction](algorithms/cs/)
- [MoDL unrolled network](algorithms/modl/)
- [Implicit neural representations for differentiable uncalibrated imaging](algorithms/implicit_repr/)

MRI physics simulation:
- [Bloch signal simulation](algorithms/bloch/)
- [Extended phase graphs (EPG) method](algorithms/epg/)

MRI quantification:
- [MRF parameter mapping using physics-informed MLP](algorithms/mrf_pm_mlp/)


## Coming Soon

Inversion:
- [Deep image prior](https://arxiv.org/abs/1711.10925)
- [Equivariant imaging](https://arxiv.org/abs/2103.14756)
- [Equivariant splitting](https://arxiv.org/abs/2510.00929)
- [Double blind imaging with generative modeling](https://arxiv.org/abs/2503.21501)
- [Compressed-sensing with generative modeling (CSGM)](https://arxiv.org/abs/1703.03208)
- Plug-and-play inversion using diffusion modeling and flow matching
- [noise2noise denoising](https://arxiv.org/abs/1803.04189)
- BM3D denoising
- [ESPIRiT parallel-imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142121/)
- [Magnetic resonance spin tomography in time-domain (MR-STAT)](https://doi.org/10.1016/j.mri.2017.10.015)
- [Data-driven discovery of mechanical models directly from MRI spectral data](https://arxiv.org/abs/2411.06958)
- [Neural Inverse Rendering from Propagating Light](https://arxiv.org/abs/2506.05347)

Representations:
- Gaussian representations
- [Structured representations using flow-based models](https://link.springer.com/book/10.1007/978-3-031-88111-4)
- [Non-linear ICA for Principled Disentanglement in Unsupervised Deep Learning](https://arxiv.org/abs/2303.16535)
- [From Pixels to Components: Eigenvector Masking for Visual Representation Learning](https://arxiv.org/abs/2502.06314)
- [Tensorial radiance fields](https://arxiv.org/abs/2203.09517)

Forward physics modeling:
- [Phase distribution graphs (PDG) method for MR signal simulation](https://pubmed.ncbi.nlm.nih.gov/38576164/)
- [Quantum mechanical MRI simulations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6641938/)
- [UltimateSynth: MRI Physics for Pan-Contrast AI](https://pubmed.ncbi.nlm.nih.gov/39713417/)
- PINNs
- Fourier neural operators



## Citation

If you use any code from this repository, please cite it as:
```
@software{Rao_Repository_of_Computational_2026,
author = {Rao, Chinmay},
month = feb,
title = {{Repository of Computational Imaging (roci)}},
version = {0.1.1},
year = {2026}
}
```