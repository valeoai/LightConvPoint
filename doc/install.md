# Installation

## Dependencies

- [Sacred](https://github.com/IDSIA/sacred)
- pymongo
- Pytorch

## Library installation

We provide two intallation modes, one that rely only on the pytorch-geometric packqge and the other that uses our c++ libraries for k-nearest neighbor search and quantized sampling.
The later is faster on our computers.


### Installation with compilation of our knn libraries
```
pip install -ve /path/to/LightConvPoint/ --install-option="--compile"
```

### Installation without compilation
```
pip install -ve /path/to/LightConvPoint/
```

