import torch
from torch import Tensor

__all__ = ["calculate_iou"]


def calculate_iou(quad_0: Tensor, quad_1: Tensor, sort_input_quads: bool = True) -> Tensor:
    """
    Calculates the Intersection over Union (IoU) between two sets of quadrilaterals.

    Parameters:
    - quad_0 (Tensor): A tensor of shape (M, 4, 2) representing the coordinates of M quadrilaterals. Each quadrilateral
      is defined by 4 vertices, and each vertex has 2 coordinates (x, y).
    - quad_1 (Tensor): A tensor of shape (N, 4, 2) representing the coordinates of N quadrilaterals. Each quadrilateral
      is defined similarly to those in quad_0.
    - sort_input_quads (bool, optional): Whether to sort the vertices of the quadrilaterals before calculating IoU.
      Sorting can be necessary depending on the specific calculation requirements. Default is True.

    Returns:
    - Tensor: An (N, M) tensor where each element [i, j] is the IoU between the i-th quadrilateral from quad_1 and the
      j-th quadrilateral from quad_0.

    This function computes the IoU for each pair of quadrilaterals from the two input tensors and returns a matrix
    of these IoU values.
    """
    if not isinstance(quad_0, Tensor): raise ValueError(f"Expected input is Tensor but got {type(quad_0)}")
    if not isinstance(quad_1, Tensor): raise ValueError(f"Expected input is a Tensor but got {type(quad_1)}")
    if not isinstance(sort_input_quads, bool): raise ValueError(f"Expected sort_input_quads to be a boolean but got {type(sort_input_quads)}")
    if quad_0.device == quad_1.device:
        return torch.ops.quad_iou.calculate_iou.default(quad_0, quad_1, sort_input_quads)
    raise ValueError(f"Expected all tensors to be on the same device but got device {quad_0.device} for quad_0 and {quad_1.device} for quad_1")
    

@torch.library.register_fake("quad_iou::calculate_iou")
def _(quad_0: Tensor, quad_1: Tensor, sort_input_quads: bool = True) -> Tensor:
    torch._check(quad_0.shape == quad_1.shape, f"Expected quad_0 and quad_1 to have the same shape but got {quad_0.shape} and {quad_1.shape}")
    torch._check(quad_0.dtype == quad_1.dtype, f"Expected quad_0 and quad_1 to have the same dtype but got {quad_0.dtype} and {quad_1.dtype}")
    torch._check(quad_0.device == quad_1.device, f"Expected quad_0 and quad_1 to be on the same device but got {quad_0.device} and {quad_1.device}")
    torch._check(quad_0.dim() == 3 and quad_0.size(1) == 4 and quad_0.size(2) == 2, f"Expected quad_0 to have shape (M, 4, 2) but got {quad_0.shape}")
    torch._check(quad_1.dim() == 3 and quad_1.size(1) == 4 and quad_1.size(2) == 2, f"Expected quad_1 to have shape (N, 4, 2) but got {quad_1.shape}")
    return torch.empty((quad_1.size(0), quad_0.size(0)), dtype=quad_0.dtype, device=quad_0.device)
