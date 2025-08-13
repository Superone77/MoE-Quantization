import torch
from .quantizer import Quantizer


class NVFP4Quantizer(Quantizer):
    """Quantizer placeholder for NVFP4 weights.

    The NVFP4 format already represents values in a floating point format
    with an implicit scale of ``1`` and zero point ``0``.  As a result the
    quantization parameters for weights are trivial and can be populated
    without inspecting the tensor statistics.
    """

    quant_type = "nvfp4"

    def find_params(self, x, weight: bool = False):
        if weight:
            # NVFP4 weights use a fixed scale of 1 and a zero point of 0
            self.scale = torch.ones((x.shape[0], 1), device=x.device)
            self.zero = torch.zeros((x.shape[0], 1), device=x.device)
            return
        # Fallback to default behaviour for non-weight tensors
        super().find_params(x, weight=weight)

