import torch
import torch.utils
import torch.utils.data
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import hyperparameters as h
import src.utils as utils
from src.dataset import Dataset
from src.YOLO import YOLO
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

utils.convert_labels(input_dir='./dataset/test/labels', output_dir='./dataset/test/annotations')
utils.convert_labels(input_dir='./dataset/valid/labels', output_dir='./dataset/valid/annotations')
utils.convert_labels(input_dir='./dataset/train/labels', output_dir='./dataset/train/annotations')

device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

transform_fn = A.Compose([
    A.LongestMaxSize(max_size=h.image_size),
    A.PadIfNeeded(min_height=h.image_size, min_width=h.image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ToGray(p=1),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo'))

train_set = torch.utils.data.DataLoader(dataset=Dataset(images_dir='./dataset/train/images', annotations_dir='./dataset/train/annotations', transform_fn=transform_fn, device=device), batch_size=h.batch_size, shuffle=True)
valid_set = torch.utils.data.DataLoader(dataset=Dataset(images_dir='./dataset/valid/images', annotations_dir='./dataset/valid/annotations', transform_fn=transform_fn, device=device), batch_size=h.batch_size, shuffle=True)

yolo = YOLO(config_filepath='./config/modules.cfg', device=device)
yolo.load_base_weights(filepath='./config/base_weights')
yolo.train(train_set=train_set, valid_set=valid_set)