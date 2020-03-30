import lightconvpoint.nn as lcp_nn


def get_conv(conv_name):
    if conv_name == "LCP":
        return lcp_nn.LCP
    elif conv_name == "ConvPoint":
        return lcp_nn.ConvPoint
    elif conv_name == "PCCN":
        return lcp_nn.PCCN
    else:
        raise Exception(f'Unknown convolution {conv_name}')


def get_search(search_name):
    if search_name == "SearchQuantized":
        return lcp_nn.SearchQuantized
    else:
        raise Exception(f'Unknown convolution {search_name}')


def get_network(model_name, in_channels, out_channels, ConvNet_name, Search_name):
    if model_name == "ConvPointSeg" or model_name == "ConvPoint":
        from lightconvpoint.networks.convpoint import ConvPointSeg as Net
    elif model_name == "ConvPointCls":
        from lightconvpoint.networks.convpoint import ConvPointCls as Net
    elif model_name == "KPConvSeg":
        from lightconvpoint.networks.kpconv import KPConvSeg as Net
    elif model_name == "KPConvCls":
        from lightconvpoint.networks.kpconv import KPConvCls as Net
    else:
        raise Exception(f'Unknown model {model_name}')

    return Net(in_channels, out_channels, get_conv(ConvNet_name), get_search(Search_name))
