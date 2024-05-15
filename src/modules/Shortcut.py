import torch

class Shortcut(torch.nn.Module):
    def __init__(self, from_idx: int) -> None:
        super(Shortcut, self).__init__()
        self.from_idx = from_idx

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        return layer_outputs[-1] + layer_outputs[self.from_idx]