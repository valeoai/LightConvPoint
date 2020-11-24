# Experiments with FKAConv with LighConvPoint framework

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

### [S3DIS](http://buildingparser.stanford.edu/dataset.html)

The S3DIS is a large indoor dataset for point cloud semantic segmentation.

**Data preparation.** We use the data preparation from ConvPoint to create the point cloud text files (https://github.com/aboulch/ConvPoint/tree/master/examples/s3dis).

### [Semantic8](http://semantic3d.net/) 

**Data preparation.** We use the data preparation from ConvPoint to create the point cloud text files (https://github.com/aboulch/ConvPoint/tree/master/examples/semantic3d) with a given voxel size.

**Benchmark file creation.** We use the projection from decimated pointcloud to original ones of ConvPoint (https://github.com/aboulch/ConvPoint/tree/master/examples/semantic3d).

### [NPM3D](https://npm3d.fr/paris-lille-3d)

**Data preparation.** The `prepare_data.py` script splits the training files into smaller files for easy loading.

### Training

To modify the settings, e.g., area, save directory... you can either modify the yaml file, create a copy of the original file with modified arguments or change the options directly in the command line.
```bash
python train.py # will automatically call the modified yaml file
python train.py with new_config_file.yaml # call the default config file and then update parameters with the new one
python train.py with area=X training.savedir="new_savedir_path" # direct modification in the command line
```
The area parameter (for S3DIS) is the test area identifier, it will train the model on every other areas.

### Test

To test the model:
```bash
python test.py -c save_directory_path/config.yaml
```

### Training and testing with a fusion model

To train a fusion model, first train independently two models with and without color information (it is an option in the config file).
Then you can train the fusion model by modifying the `s3dis_fusion.yaml` file (mostly set the paths of the two previously trained models) and:
```
python train.py with s3dis_fusion.yaml
```
The test is the same as previous:
```bash
python test.py -c save_directory_path/config.yaml
```

#### Pretrained models

Pretrained models are available for [S3DIS](https://github.com/valeoai/LightConvPoint/releases/download/v0.1/s3dis_KPConvSeg.zip), [Semantic8](https://github.com/valeoai/LightConvPoint/releases/download/v0.1/semantic8_KPConvSeg.zip) and [NPM3D](https://github.com/valeoai/LightConvPoint/releases/download/v0.1/npm3d_KPConvSeg.zip).
