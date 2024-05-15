import torch

class Route(torch.nn.Module):
    def __init__(self, from_idx: int, to_idx: int) -> None:
        super(Route, self).__init__()

        self.from_idx = from_idx
        self.to_idx = to_idx

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        from_layer_output = layer_outputs[self.from_idx]
        if self.to_idx == 0:
            return from_layer_output
        
        to_layer_output = layer_outputs[self.to_idx]
        return torch.concat((from_layer_output, to_layer_output), dim=1)