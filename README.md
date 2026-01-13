# Repository of Computational Imaging

Self-contained implementations of some algorithms used in computational imaging, focused especially on MRI. This repo has a simple structure: `algorithm = python_file + demo_notebook`.
```
algorithms
    |
    |- algo_1
    |   |- algo_1.py
    |   |- Demo.ipynb
    |
    |- algo_2
    |   |- algo_2.py
    |   |- Demo.ipynb
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
