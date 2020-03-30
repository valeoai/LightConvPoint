# Getting started

LightConvPoint is aimed to a simple use.

## Convolutional layer

The convolution layer ```Conv``` of the ```nn``` subpackage is a container class in wich can be plugged various convolutional layers and neaighborhood search algorithms.

For example to create a convolutional layer a the LCP formulation with quantized search algorithm:

```python
# import the nn package of LightConvpoint 
import lightconvpoint.nn as lcp_nn

# define a convolutional layer with
# - LCP formulation (32 input channels, 64 output channels and 16 kernel elements)
# - QuantizedSearch (16 neighbors, stride 2)
conv_layer = lcp_nn.Conv(lcp_nn.LCP(32, 64, 16), lcp_nn.QuantizedSearch(K=16, stride=2))

input_pts = torch.rand(5, 3, 1024) # Batchsize 5, dimension 3, num. points 1024
input = torch.rand(5, 32, 1024) # features associated with points

output, output_pts, indices = conv_layer(input, input_pts)

print(outputs_pts.shape) # (5, 3, 512)
print(output.shape) # (5, 64, 512)
```

## A first network
