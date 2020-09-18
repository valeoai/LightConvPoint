import torch
import torch.nn as nn


class Conv(nn.Module):
    """Meta convolution layer.

    Core component of the convolutional neural networks build with LightConvPoint.

    # Arguments
        network: network object.
            Instance of a convolutional layer.
        search: search object.
            Instance of a search object.
        activation: (optional) activation layer.
            Add an activation layer integrated to the convolutional layer
        normalization: (optional) 1-D normalization layer.
            Add a normalization layer integrated to the convolutional layer
        bn_activation_order: batch normalization and activation order
            Define the operation order, default is BN->Activation ("bn_act"), can be
            "act_bn" for Activation->BN

    # Forward arguments
        input: 3-D torch tensor.
            Input feature tensor. Dimensions are (B, I, N) with B the batch size, I the
            number of input channels and N the number of input points.
        points: 3-D torch tensor.
            The input points. Dimensions are (B, D, N) with B the batch size, D the
            dimension of the spatial space and N the number of input points.
        support_points: (optional) 3-D torch tensor.
            The support points to project features on. If not provided, use the `search`
            object of the layer to compute them.
            Dimensions are (B, D, N) with B the batch size, D the dimenstion of the
            spatial space and N the number of input points.
        indices: (optional) 3-D torch tensor.
            The indices of the neighboring points with respect to the support points.
            If not provided, use the `search` object of the layer to compute them.

    # Forward returns
        features: 3-D torch tensor.
            The computed features. Dimensions are (B, O, N) with B the batch size, O the
            number of output channels and N the number of input points.
        support_points: 3-D torch tensor.
            The support points. If they were provided as an input, return the same
            tensor.
        indices: 3-D torch tensor.
            The indices of the neighboring points with respect to the support points.
            If they were provided as an input, return the same tensor.
    """

    def __init__(
        self, network, search, activation=None, normalization=None,
        bn_activation_order="bn_act"
        ):
        super().__init__()
        self.network = network
        self.search = search
        self.activation = activation
        self.norm = normalization
        self.bn_activation_order = bn_activation_order

    def batched_index_select(self, input, dim, index):
        """Gather input with respect to the index tensor."""
        index_shape = index.shape
        views = [input.shape[0]] + [
            1 if i != dim else -1 for i in range(1, len(input.shape))
        ]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index).view(
            input.size(0), -1, index_shape[1], index_shape[2]
        )

    def forward(self, input, points, support_points=None, indices=None):
        """Forward function of the layer."""

        if indices is None:
            # search the support points and the neighborhoods
            indices, support_points = self.search(points, support_points)

        if input is None:
            # inpuy is None: do not compute features
            return None, support_points, indices
        else:
            # compute the features
            indices = indices.clone()

            # get the features and point coordinates associated with the indices
            pts = self.batched_index_select(points, dim=2, index=indices).contiguous()
            features = self.batched_index_select(
                input, dim=2, index=indices
            ).contiguous()

            # predict the features
            features, support_points = self.network(
                features, pts, support_points.contiguous()
            )

            if self.bn_activation_order=="act_bn":
                # apply activation
                if self.activation is not None:
                    features = self.activation(features)

                # apply normalization
                if self.norm is not None:
                    features = self.norm(features)

            else: #apply default BN->ACT
                # apply normalization
                if self.norm is not None:
                    features = self.norm(features)

                # apply activation
                if self.activation is not None:
                    features = self.activation(features)

            return features, support_points, indices
