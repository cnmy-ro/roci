<p align="center">
  <img src="docs/roci_logo.png"  width="400">
</p>

[![DOI](https://zenodo.org/badge/1132861288.svg)](https://doi.org/10.5281/zenodo.18778617)

Repository of computational imaging (`roci`) is a collection of clean, self-contained implementations of algorithms used in computational imaging. Currently focused on computational MRI applications (reconstruction, synthesis, quantification) and techniques based on representation learning and generative modeling.

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
- [SENSE reconstruction](algorithms/sense/)
- [CG-SENSE reconstruction](algorithms/cg_sense/)
- [Compressed sensing reconstruction](algorithms/cs/)

MR signal simulation:
- [Bloch simulation](algorithms/bloch/)
- [Extended phase graph method](algorithms/epg/)

## Coming Soon

- [ ] Deep image prior
- [ ] Compressed-sensing with generative modeling (CSGM)
- [ ] Diffusion model-based reconstruction
- [ ] Flow matching-based reconstruction
- [ ] Implicit neural representations
- [ ] Gaussian splatting
- [ ] Plug-and-play denoiser-based reconstruction
- [ ] Magnetic resonance spin tomography in time-domain (MR-STAT)