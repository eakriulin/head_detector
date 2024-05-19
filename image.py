import sys
import os
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import hyperparameters as h
from src.YOLO import YOLO
from src.utils import draw_circles

transform_fn = A.Compose([
    A.LongestMaxSize(max_size=h.image_size),
    A.PadIfNeeded(min_height=h.image_size, min_width=h.image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ToGray(p=1),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ToTensorV2(),
])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Provide the path to the input image, e.g., python image.py /path/to/image.png')

    input_filepath = sys.argv[1]

    if not os.path.exists(input_filepath):
        sys.exit('There is no image at this path')

    dirpath, input_filename = os.path.split(input_filepath)
    output_filepath = os.path.join(dirpath, f'processed_{input_filename}')

    image = cv2.imread(input_filepath)
    transformed = transform_fn(image=image)
    transformed_image: torch.Tensor = transformed['image']

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(config_filepath='./config/modules.cfg', device=device)
    yolo.load(yolo.best_valid_weights_filepath)

    input = transformed_image.unsqueeze(0)
    output = yolo.detect(input)

    if len(output) == 0:
        sys.exit('Sorry, nothing was detected')

    output_image = draw_circles(image, output)
    cv2.imwrite(output_filepath, output_image)

    print(f'Output image: {output_filepath}')