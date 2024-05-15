import torch
import torch.nn.functional as F

class Loss():
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        object_mask = target[..., 3] == 1
        no_object_mask = target[..., 3] == 0

        output_boxes = torch.sigmoid(output[..., :3])
        output_object = output[..., 3]

        target_boxes = target[..., :3]
        target_object = target[..., 3]

        no_object_loss = F.binary_cross_entropy_with_logits(output_object[no_object_mask], target_object[no_object_mask], reduction='sum')
        object_loss = F.binary_cross_entropy_with_logits(output_object[object_mask], target_object[object_mask], reduction='sum')
        boxes_loss = F.mse_loss(output_boxes[object_mask], target_boxes[object_mask], reduction='sum')

        loss = boxes_loss + object_loss + no_object_loss
        return loss / output.size(0)
  