# Installation

## Dependencies

- [Sacred](https://github.com/IDSIA/sacred)
- pymongo
- Pytorch

## Library installation

We provide two intallation modes, one that rely only on the pytorch-geometric packqge and the other that uses our c++ libraries for k-nearest neighbor search and quantized sampling.
The later is faster on our computers.

### Installation with compilation of our knn libraries

This is the default installation procedure, and the one used in the FKAConv paper.

```
pip install -ve /path/to/LightConvPoint/
```

### Installation without compilation

We also provide a version that rely only on pytorch geometric.
However, this version is slower and comes with no performance guaranty.

```
pip install -ve /path/to/LightConvPoint/ --install-option="--nocompile"
```
