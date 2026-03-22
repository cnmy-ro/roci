<p align="center">
  <img src="docs/roci_logo.png"  width="400">
</p>

[![DOI](https://zenodo.org/badge/1132861288.svg)](https://doi.org/10.5281/zenodo.18778617)

Repository of computational imaging (`roci`) is a collection of clean, self-contained implementations of algorithms used in computational imaging. Currently focused on computational MRI applications (reconstruction, synthesis, quantification) and techniques based on representation learning and generative modeling. For educational and research purposes only.

## Organization

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

MRI physics simulation:
- [Bloch signal simulation](algorithms/bloch/)
- [Extended phase graphs method](algorithms/epg/)

Image representations:
- [Implicit neural representations for differentiable uncalibrated imaging](algorithms/implicit_repr/)


## Coming Soon

Inversion:
- [ ] Deep image prior
- [ ] Double Blind Imaging with Generative Modeling
- [ ] Compressed-sensing with generative modeling (CSGM)
- [ ] Diffusion model-based inversion
- [ ] Flow matching-based inversion
- [ ] noise2noise denoising
- [ ] BM3D denoising
- [ ] Plug-and-play denoiser-based reconstruction
- [ ] MoDL
- [ ] ESPIRiT parallel-imaging
- [ ] Magnetic resonance spin tomography in time-domain (MR-STAT)

Representations and physics:
- [ ] PINNs
- [ ] Neural operators
- [ ] Gaussian representations
- [ ] Equivariant imaging
