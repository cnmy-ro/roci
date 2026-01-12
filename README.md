# Computaional Imaging

Self-contained implementations of some algorithms used in computational imaging, especially MRI. This repo has a simple structure: `algorithm = python_file + demo_notebook`.
```
algorithms
    |
    |- algorithm_1
    |   |- algorithm_1.py
    |   |- Demo.ipynb
    |
    |- algorithm_2
    |   |- algorithm_2.py
    |   |- Demo.ipynb
    ...
```


## Available Algorithms

MRI reconstruction:
- [SENSE reconstruction](algorithms/sense/)
- [CG-SENSE reconstruction](algorithms/cg_sense/)
- [Compressed sensing reconstruction](algorithms/cs/)

MR signal simulation:
- [Extended phase graph method](algorithms/epg/)