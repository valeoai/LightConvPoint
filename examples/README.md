# Experiments with LCP

All the parameters can be changed either in the JSON file associated with the example or with ```with``` statement from Sacred.
For example, changing the save directory and the data directory would become:
```
python train.py with file.json datasetdir="new/path/to/datatset" savedir="/new/path/to/save/directory"
```

## Classification with ModelNet40

We use the HDF5 file from [https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).

```
cd modelnet40
python train.py with modelnet40.json
```

## Part segmentation

```
cd shapenet
python train.py with shapenet.json
```

## Semantic Segmentation

*To be released*
