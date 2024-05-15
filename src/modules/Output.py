import torch
import hyperparameters as h

class Output(torch.nn.Module):
    def __init__(self):
        super(Output, self).__init__()

    def forward(self, input: torch.Tensor):
        batch_size, _, x, y = input.shape

        input = input.view(batch_size, h.n_of_anchors * h.n_of_attributes_per_box, x * y)
        input = input.transpose(1, 2).contiguous()
        input = input.view(batch_size, x * y * h.n_of_anchors, h.n_of_attributes_per_box)

        return input