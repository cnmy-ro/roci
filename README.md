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
- [ ] [Deep image prior](https://arxiv.org/abs/1711.10925)
- [ ] [Equivariant imaging](https://arxiv.org/abs/2103.14756)
- [ ] [Double Blind Imaging with Generative Modeling](https://arxiv.org/abs/2503.21501)
- [ ] [Compressed-sensing with generative modeling (CSGM)](https://arxiv.org/abs/1703.03208)
- [ ] Diffusion model-based inversion
- [ ] Flow matching-based inversion
- [ ] [noise2noise denoising](https://arxiv.org/abs/1803.04189)
- [ ] BM3D denoising
- [ ] Plug-and-play denoiser-based reconstruction
- [ ] [MoDL](https://arxiv.org/abs/1712.02862)
- [ ] [ESPIRiT parallel-imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142121/)
- [ ] [Magnetic resonance spin tomography in time-domain (MR-STAT)](https://doi.org/10.1016/j.mri.2017.10.015)

Representations and physics:
- [ ] PINNs
- [ ] Neural operators
- [ ] Gaussian representations
- [ ] [Structured and disentangled representations using flow-based models](https://link.springer.com/book/10.1007/978-3-031-88111-4)
