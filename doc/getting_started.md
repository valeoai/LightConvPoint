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

We can now build a simple network for classification with 3 convolutions and 2 linear layers.
Each convolution will use a number of support points of 1/4 the size of the input point cloud.

```python
import lightconvpoint.nn as lcp_nn

class ClassificationNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, ConvNet, Search):
        super().__init__()

        self.cv1 = lcp_nn.Conv(
            ConvNet(in_channels, 64, 16),
            Search(K=16, stride=4),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(64),
        )
        self.cv2 = lcp_nn.Conv(
            ConvNet(64, 128, 16),
            Search(K=16, stride=4),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(128),
        )
        self.cv3 = lcp_nn.Conv(
            ConvNet(128, 256, 16),
            Search(K=16, stride=4),
            activation=nn.ReLU(),
            normalization=nn.BatchNorm1d(256),
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linera(128, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, input_pts):

        x1, pts1, ids1 = self.cv1(x, input_pts)
        x2, pts2, ids2 = self.cv2(x1, pts1)
        x3, pts3, ids3 = self.cv3(x2, pts2)

        xout = x3.mean(2) # average over remaining points
        xout = self.relu(self.fc1(xout))
        xout = self.fc2(xout)

        return xout
```

## Accelerating the computation

*TODO*