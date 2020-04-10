# [LightConvPoint](https://arxiv.org/abs/2004.04462)

A framework to build convolutional network for point cloud processing.

![SnapNet products](./doc/predictions.png)

## Paper

The paper is available at arxiv: [https://arxiv.org/abs/2004.04462](https://arxiv.org/abs/2004.04462)

If you use this code in your research, please consider citing:

```
@article{boulch2020lightconvpoint,
  title={LightConvPoint: convolution for points},
  author={Boulch, Alexandre and Puy, Gilles and Marlet, Renaud},
  journal={arXiv preprint arXiv:2004.04462},
  year={2020}
}
```

## Ressources

* [Installation](doc/install.md): install and setup lightconvpoint
* [Run experients](examples/README.md): re-run experiments from the paper
* [Getting started](doc/getting_started.md): start to design your own network
* [Library features and implemented algorithms](doc/features.md): description of avalailable algorithms in LCP (convolutional layers such as LightConvPoint or ConvPoint; support point selection including quantized search or farthest point sampling).

### Example

We provide examples classification and segmentation datasets:
* ModelNet40
* ShapeNet
* S3DIS (*to be released*)
* Semantic8 (*to be released*)
* NPM3D (*to be released*)

