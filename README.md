# ROCI: Repository of Computational Imaging

<p align="center">
  <img src="docs/compimg_graphic.png"  width="800" alt="This figure uses the Calabi-Yau manifold SVG as a component, taken from: https://commons.wikimedia.org/wiki/File:Calabi_yau_formatted.svg">
</p>

[![DOI](https://zenodo.org/badge/1132861288.svg)](https://doi.org/10.5281/zenodo.18778617)

ROCI is a collection of minimalistic, high-quality, and self-contained PyTorch implementations of computational imaging algorithms. 

In terms of applications, ROCI focuses on forward and inverse problems appearing in computational MRI (reconstruction, quantification, synthesis, physics simulations). In terms of techniques, it provides classical signal processing algorithms alongside SoTA representation learning and generative modeling methods. All algorithms are written in PyTorch. This means autograd compatibility and GPU-accelerated compute. 


<!-- ## Why ROCI

Original software implmentations offered by paper authors vary largely in quality and style and may be incomplete, outdated, or implemented in a less familiar language or library. This introduces a barrier for other applied reseachers benchmarking their work, engineers building prototype systems on these techniques, or students learning practical conventions and tricks. ROCI bridges this gap by offering software implementations that are concise, self-contained, standardized in style, and well-documented. 

These implementations are tested for correctness, are based primarily on PyTorch, and follow the PEP8 style guide. This means a wide range of readily accessible and usable computational imaging algorithms, all in one place. -->


## Code Organization

All algorithms are stored in the [`algorithms`](algorithms) directory. The directory structure is simple: `algorithm = python_file + demo_notebook + readme`.
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

We also provide tiny versions of some algorithms applied on miniature toy problems, e.g. generative modeling of toy distributions in 2D space. These can be found in [`tiny`](tiny).


## Available Algorithms

MRI reconstruction:
- [SENSE parallel-imaging reconstruction](algorithms/sense/)
- [CG-SENSE parallel-imaging reconstruction](algorithms/cg_sense/)
- [Compressed sensing reconstruction](algorithms/cs/)
- [GRAPPA parallel-imaging reconstruction](algorithms/grappa/)
- [ESPIRiT parallel-imaging calibration](algorithms/espirit/)
- [PnP-CNN](algorithms/pnp_cnn/)
- [MoDL unrolled network](algorithms/modl/)
- [Implicit neural representations for differentiable uncalibrated imaging](algorithms/diff_uncalib_img/)
- [Reconstruction with deep image prior](algorithms/dip/)

MRI forward physics simulation:
- [Bloch signal simulation](algorithms/bloch/)
- [Extended phase graphs (EPG) method](algorithms/epg/)
- [UltimateSynth: MRI Physics for Pan-Contrast AI](https://pubmed.ncbi.nlm.nih.gov/39713417/)

MRI quantification:
- [Physics-informed MLP for MRF parameter mapping](algorithms/mrf_pm_mlp/)


## Coming Soon

Inversion:
- [Implicit ESPIRiT: A compact, implicit representation of ESPIRiT maps with stochastic learning of eigenvectors](https://echo.ismrm.org/abstracts/view/f8ca3f28-6917-463e-b64c-66194f7ec8ab)
- [Equivariant imaging](https://arxiv.org/abs/2103.14756)
- [Equivariant splitting](https://arxiv.org/abs/2510.00929)
- [Double blind imaging with generative modeling](https://arxiv.org/abs/2503.21501)
- [Compressed-sensing with generative modeling (CSGM)](https://arxiv.org/abs/1703.03208)
- [PnP-CoSMo: Plug-and-play reconstruction based on content/style modeling](https://doi.org/10.1016/j.media.2026.104160)
- [PnP-Flow](https://proceedings.iclr.cc/paper_files/paper/2025/hash/708e58b0b99e3e62d42022b4564bad7a-Abstract-Conference.html)
- [NullFlow: One-Step Generative Reconstruction](https://arxiv.org/abs/2606.22696)
- [PCFlow: Perceptually Consistent Flow Matching for Efficient Image Restoration](https://arxiv.org/html/2608.10544v1)
- [PnP-CM](https://arxiv.org/abs/2509.22736)
- [Regularization by Denoising (RED)](https://epubs.siam.org/doi/10.1137/16M1102884)
  - [Consistency Models for Fast MRI Reconstruction Using Regularization by Denoising](https://arxiv.org/abs/2608.20561)
- [i-DEQ: A stable inertial deep equilibrium model for image restoration](https://arxiv.org/abs/2608.10001)
- [noise2noise denoising](https://arxiv.org/abs/1803.04189)
- [BM3D denoising](https://ieeexplore.ieee.org/document/4271520)
- [Motion-corrected MRI with DISORDER](https://pubmed.ncbi.nlm.nih.gov/31898832/)
- [Magnetic resonance spin tomography in time-domain (MR-STAT)](https://doi.org/10.1016/j.mri.2017.10.015)
- [Data-driven discovery of mechanical models directly from MRI spectral data](https://arxiv.org/abs/2411.06958)
- [SIINR: structurally informed implicit neural representations](https://arxiv.org/abs/2607.19943)
- [Neural Inverse Rendering from Propagating Light](https://arxiv.org/abs/2506.05347)
- [Exoplanet Imaging via Differentiable Rendering](https://ieeexplore.ieee.org/document/10824793)

Representations:
- Gaussian representations
- [Structured representations using flow-based models](https://link.springer.com/book/10.1007/978-3-031-88111-4)
- [Non-linear ICA for Principled Disentanglement in Unsupervised Deep Learning](https://arxiv.org/abs/2303.16535)
- [From Pixels to Components: Eigenvector Masking for Visual Representation Learning](https://arxiv.org/abs/2502.06314)
- [Tensorial radiance fields](https://arxiv.org/abs/2203.09517)
- JEPA-based models

Forward physics modeling:
- [Unified Bloch and EPG method](https://www.nature.com/articles/s41598-021-00233-6)
- [Phase distribution graphs (PDG) method for MR signal simulation](https://pubmed.ncbi.nlm.nih.gov/38576164/)
- [Quantum mechanical MRI simulations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6641938/)
- [Fast and accurate Bloch simulations using Magnus expansions](https://echo.ismrm.org/abstracts/view/4b3df25a-d77c-4462-99fa-83b161b39689)
- [Neural surrogates based on PINNs](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Neural surrogates based on Fourier neural operators](https://arxiv.org/abs/2010.08895)


## Citation

If you use any code from this repository, please cite it as:
```BibTeX
@software{Rao_ROCI_2026,
  author = {Rao, Chinmay},
  month = feb,
  title = {{Repository of Computational Imaging (ROCI)}},
  version = {0.1.1},
  year = {2026},
  doi = {10.5281/zenodo.18842184},
  url = {https://doi.org/10.5281/zenodo.18842184}
}
```