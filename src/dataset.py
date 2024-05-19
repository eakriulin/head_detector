import os
import torch
import torch.utils.data
import numpy as np
import cv2
import albumentations as A
import hyperparameters as h

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, annotations_dir: str, transform_fn: A.Compose, device: torch.device) -> None:
        self.image_filepaths: list[str] = [os.path.join(images_dir, filename[:-3] + 'jpg') for filename in os.listdir(annotations_dir)]
        self.annotation_filepaths: list[str] = [os.path.join(annotations_dir, filename) for filename in os.listdir(annotations_dir)]
        self.image_filepaths.sort()
        self.annotation_filepaths.sort()

        self.transform_fn = transform_fn
        self.device = device

    def __len__(self) -> int:
        return len(self.image_filepaths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            image_filepath = self.image_filepaths[idx]
            annotation_filepath = self.annotation_filepaths[idx]

            image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
            boxes = np.roll(np.loadtxt(annotation_filepath, delimiter=' ', ndmin=2, dtype=np.float32), shift=4).tolist()

            transformed = self.transform_fn(image=image, bboxes=boxes)

            image: torch.Tensor = transformed['image'].to(device=self.device)
            boxes = torch.tensor(transformed['bboxes']).float()[..., :3] # note: keeping only (x, y, d)
            boxes = boxes.to(device=self.device)

            anchors = torch.tensor(h.anchors[0] + h.anchors[1] + h.anchors[2]).to(device=self.device)
            targets = [torch.zeros((grid_size, grid_size, h.n_of_anchors, h.n_of_attributes_per_box)) for grid_size in h.grid_sizes]
            targets = [target.to(device=self.device) for target in targets]

            for box in boxes:
                iou_with_anchors = self.iou(box[2], anchors)
                anchor_indices = iou_with_anchors.argsort(descending=True, dim=0)

                x, y, d = box

                for anchor_idx in anchor_indices:
                    grid_idx = anchor_idx // h.n_of_anchors
                    grid_anchor_idx = anchor_idx % h.n_of_anchors

                    grid_size = h.grid_sizes[grid_idx]
                    cell_x = int(grid_size * x)
                    cell_y = int(grid_size * y)

                    is_taken = bool(targets[grid_idx][cell_x, cell_y, grid_anchor_idx, 3])
                    if is_taken:
                        continue

                    bx = grid_size * x - cell_x
                    by = grid_size * y - cell_y
                    targets[grid_idx][cell_x, cell_y, grid_anchor_idx] = torch.tensor([bx, by, d, 1])

                    break

            targets = [target.view(target.size(0) * target.size(0) * h.n_of_anchors, h.n_of_attributes_per_box) for target in targets]

        return image, targets

    def iou(self, diameter1: torch.Tensor, diameter2: torch.Tensor) -> torch.Tensor:
        r1 = diameter1 / 2.0
        r2 = diameter2 / 2.0
        
        area1 = torch.pi * r1 ** 2
        area2 = torch.pi * r2 ** 2
        
        area_intersection = torch.min(area1, area2)
        area_union = torch.max(area1, area2)
        
        iou = area_intersection / area_union
        return iou