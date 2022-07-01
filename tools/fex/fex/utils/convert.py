""" convert network """

import torch

def convert_network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    def convert_module(module, dtype):
        """
        Converts a module's immediate parameters and buffers to dtype.
        """
        for param in module.parameters(recurse=False):
            if param is not None:
                if param.data.dtype.is_floating_point:
                    param.data = param.data.to(dtype=dtype)
                if param._grad is not None and param._grad.data.dtype.is_floating_point:
                    param._grad.data = param._grad.data.to(dtype=dtype)
        for buf in module.buffers(recurse=False):
            if buf is not None and buf.data.dtype.is_floating_point:
                buf.data = buf.data.to(dtype=dtype)
    for module in network.modules():
        # if isinstance(module, BertLayerNorm):
        #     continue
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype=torch.half)
    return network