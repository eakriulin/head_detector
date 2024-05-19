import os
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import numpy as np
import hyperparameters as h
from src.modules.Shortcut import Shortcut
from src.modules.Route import Route
from src.modules.Output import Output

class YOLO():
    def __init__(self, config_filepath: str, device: torch.device) -> None:
        super(YOLO, self).__init__()

        self.best_valid_weights_filepath = os.path.join('best_valid_weights.pth')
        self.best_train_weights_filepath = os.path.join('best_train_weights.pth')
        self.last_train_weights_filepath = os.path.join('last_train_weights.pth')
        self.device = device

        self.blocks = self._read_config(config_filepath)
        self.neural_network = self._create_modules(self.blocks).to(device=self.device)

    def save(self, filepath: str) -> None:
        torch.save(self.neural_network.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.neural_network.load_state_dict(torch.load(filepath, map_location=self.device))

    def train(self, train_set: torch.utils.data.DataLoader, valid_set: torch.utils.data.DataLoader) -> None:
        self.neural_network.train()
        optimizer = torch.optim.Adam(self.neural_network.parameters(), h.learning_rate)

        anchors = torch.tensor(h.anchors).to(device=self.device)

        best_train_loss = float('inf')
        best_valid_ap = float('-inf')

        for e in range(0, h.n_of_epochs):
            epoch_loss = 0
            epoch_n_of_examples = 0

            for batch, targets in train_set:
                optimizer.zero_grad()

                outputs = self.forward(batch)

                batch_loss = [self.calculate_loss(outputs[i], targets[i], anchors[i], h.grid_sizes[i]) for i in range(len(outputs))]
                batch_loss = torch.sum(torch.stack(batch_loss))

                batch_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    batch_n_of_examples = len(batch)
                    epoch_loss += batch_loss.item() * batch_n_of_examples
                    epoch_n_of_examples += batch_n_of_examples

                    print(f'\tbatch: train loss {batch_loss.item():4f}')

            epoch_loss /= epoch_n_of_examples
            if epoch_loss <= best_train_loss:
                best_train_loss = epoch_loss
                self.save(filepath=self.best_train_weights_filepath)

            epoch_valid_ap = self.eval(valid_set)
            if epoch_valid_ap >= best_valid_ap:
                best_valid_ap = epoch_valid_ap
                self.save(filepath=self.best_valid_weights_filepath)

            print(f'\nepoch {e + 1} — train loss: {epoch_loss:4f}, valid AP: {epoch_valid_ap:4f} | best valid AP: {best_valid_ap:4f}')
            self.save(filepath=self.last_train_weights_filepath)

    def eval(self, valid_set: torch.utils.data.DataLoader) -> float:
        with torch.no_grad():
            has_been_in_train_mode = self.neural_network.training
            self.neural_network.eval()

            anchors = torch.tensor(h.anchors).to(device=self.device)

            all_targets: torch.Tensor | None = None
            all_outputs: torch.Tensor | None = None

            for batch, targets in valid_set:
                targets = [self.scale(targets[i], h.grid_sizes[i]) for i in range(len(targets))]
                targets = torch.concat(targets, dim=1)

                outputs = self.forward(batch)

                outputs = [self.activate_and_scale(outputs[i], anchors[i], h.grid_sizes[i]) for i in range(len(outputs))]
                outputs = torch.concat(outputs, dim=1)

                all_targets = targets if all_targets is None else torch.concat((all_targets, targets), dim=0)
                all_outputs = outputs if all_outputs is None else torch.concat((all_outputs, outputs), dim=0)

            ap = self.calculate_average_precision(all_outputs, all_targets)

            if has_been_in_train_mode:
                self.neural_network.train()

        return ap.item()
    
    def detect(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.neural_network.eval()
            return self.non_max_suppression(self.forward(input))

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        blocks = self.blocks[1:]
        layer_outputs: list[torch.Tensor] = []
        yolo_outputs: list[torch.Tensor] = []

        for idx, block in enumerate(blocks):
            layer: torch.nn.Module = self.neural_network[idx]
            layer_type: str = block['type']

            if layer_type in ['convolutional', 'upsample']:
                input: torch.Tensor = layer(input)

            if layer_type in ['shortcut', 'route']:
                input: torch.Tensor = layer(layer_outputs)

            if layer_type == 'yolo':
                input: torch.Tensor = layer(input)
                yolo_outputs.append(input)

            layer_outputs.append(input)

        return yolo_outputs
    
    def calculate_loss(self, output: torch.Tensor, target: torch.Tensor, anchors: torch.Tensor, grid_size: int) -> torch.Tensor:
        object_mask = target[..., 3] == 1
        no_object_mask = target[..., 3] == 0

        anchors = anchors.repeat(1, grid_size * grid_size)

        output_coordinates = torch.sigmoid(output[..., :2])
        output_box = torch.exp(output[..., 2]) * anchors
        output_object = output[..., 3]

        target_coordinates = target[..., :2]
        target_box = target[..., 2]
        target_object = target[..., 3]

        no_object_loss = F.binary_cross_entropy_with_logits(output_object[no_object_mask], target_object[no_object_mask], reduction='sum')
        object_loss = F.binary_cross_entropy_with_logits(output_object[object_mask], target_object[object_mask], reduction='sum')
        coordinates_loss = F.mse_loss(output_coordinates[object_mask], target_coordinates[object_mask], reduction='sum')
        box_loss = F.mse_loss(output_box[object_mask], target_box[object_mask], reduction='sum')

        loss = no_object_loss + object_loss + coordinates_loss + box_loss
        return loss / output.size(0)
        
    def calculate_average_precision(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # note: lists of True/False Positives
        tp = torch.tensor([]).to(device=self.device)
        fp = torch.tensor([]).to(device=self.device)

        for output_idx, output in enumerate(outputs):
            sorted_output_indices = torch.sort(output[..., 3], descending=True)[1]
            output = output[sorted_output_indices]

            target = targets[output_idx]
            used_target_indices = [False] * target.size(0)

            for detection in output:
                if detection[..., 3] < h.confidence_threshold:
                    continue

                ious = self.iou(detection.unsqueeze(0), target).squeeze()
                sorted_iou_indices = torch.sort(ious, descending=True)[1]

                max_iou = float('-inf')
                max_target_idx = -1

                for target_idx in sorted_iou_indices:
                    if used_target_indices[target_idx]:
                        continue

                    max_iou = ious[target_idx].item()
                    max_target_idx = target_idx
                    break

                if max_iou >= h.iou_threshold:
                    tp = torch.concat((tp, torch.tensor([1.0]).to(device=self.device)), dim=0)
                    fp = torch.concat((fp, torch.tensor([0.0]).to(device=self.device)), dim=0)
                else:
                    fp = torch.concat((fp, torch.tensor([1.0]).to(device=self.device)), dim=0)
                    tp = torch.concat((tp, torch.tensor([0.0]).to(device=self.device)), dim=0)

                used_target_indices[max_target_idx] = True

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        recall = tp_cumsum / len(targets)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = torch.trapz(precision, recall)
        return ap

    def non_max_suppression(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        anchors = torch.tensor(h.anchors).to(device=self.device)

        detections = [self.activate_and_scale(outputs[i], anchors[i], h.grid_sizes[i]) for i in range(len(outputs))]
        detections = torch.concat(detections, dim=1)

        confidence_mask = (detections[..., 3] >= h.confidence_threshold).float().unsqueeze(2)
        detections = detections * confidence_mask

        batch_size = detections.size(0)
        output = [None] * batch_size

        for image_idx, image_detections in enumerate(detections):
            non_zero_indices = torch.nonzero(image_detections[..., 3]).flatten()
            if len(non_zero_indices) == 0:
                continue

            image_detections = image_detections[non_zero_indices]
            sorted_indices = torch.sort(image_detections[..., 3], descending=True)[1]
            image_detections = image_detections[sorted_indices]

            box_idx = 0
            while box_idx < image_detections.size(0):
                ious = self.iou(image_detections[box_idx].unsqueeze(0), image_detections[box_idx + 1:])
                iou_mask = (ious < h.iou_threshold).float().unsqueeze(1)
                image_detections[box_idx + 1:] *= iou_mask

                non_zero_indices = torch.nonzero(image_detections[..., 3]).flatten()
                image_detections = image_detections[non_zero_indices]
                box_idx += 1

            output[image_idx] = image_detections if output[image_idx] is None else torch.concat((output[image_idx], image_detections))

        output = [image_detections for image_detections in output if image_detections is not None]
        return torch.stack(output) if len(output) > 0 else torch.tensor([]).to(device=self.device)
    
    def iou(self, box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        x1, y1, d1 = box[..., 0], box[..., 1], box[..., 2]
        r1 = d1 / 2

        x2, y2, d2 = boxes[..., 0], boxes[..., 1], boxes[..., 2]
        r2 = d2 / 2

        dist = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        area1 = torch.pi * r1 ** 2
        area2 = torch.pi * r2 ** 2

        no_overlap = dist >= (r1 + r2)
        completely_within = dist <= torch.abs(r1 - r2)

        term1 = r1 ** 2 * torch.acos((dist**2 + r1 ** 2 - r2 ** 2) / (2 * dist * r1)).clamp(min=0)
        term2 = r2 ** 2 * torch.acos((dist**2 + r2 ** 2 - r1 ** 2) / (2 * dist * r2)).clamp(min=0)
        term3 = 0.5 * torch.sqrt((-dist + r1 + r2).clamp(min=0) * (dist + r1 - r2).clamp(min=0) * (dist - r1 + r2).clamp(min=0) * (dist + r1 + r2).clamp(min=0))
        area_of_intersection = term1 + term2 - term3

        area_of_intersection[no_overlap] = 0.0
        if completely_within.any():
            area_of_intersection[completely_within] = torch.pi * torch.min(r1, r2[completely_within]) ** 2

        union_area = area1 + area2 - area_of_intersection

        ious = area_of_intersection / union_area
        ious[no_overlap] = 0.0

        return ious

    def activate_and_scale(self, output: torch.Tensor, anchors: torch.Tensor, grid_size: int) -> torch.Tensor:
        return self.scale(self.activate(output, anchors, grid_size), grid_size)

    def activate(self, output: torch.Tensor, anchors: torch.Tensor, grid_size: int) -> torch.Tensor:
        anchors = anchors.repeat(1, grid_size * grid_size)

        output[..., 0] = torch.sigmoid(output[..., 0]) # note: activation of x
        output[..., 1] = torch.sigmoid(output[..., 1]) # note: activation of y
        output[..., 2] = torch.exp(output[..., 2]) * anchors # note: activation of d
        output[..., 3] = torch.sigmoid(output[..., 3]) # note: activation of object confidence

        return output

    def scale(self, output: torch.Tensor, grid_size: int) -> torch.Tensor:
        # note: applying offset for x and y
        grid = torch.arange(grid_size).float()
        grid_x, grid_y = torch.meshgrid(grid, grid, indexing='xy')
        x_offset = grid_x.reshape(-1, 1)
        y_offset = grid_y.reshape(-1, 1)
        offsets = torch.concat((y_offset, x_offset), dim=1).repeat(1, h.n_of_anchors).view(-1, 2)
        offsets = offsets.to(device=self.device)
        output[..., :2] += offsets

        # note: scaling from grid size to image size
        stride = h.image_size / grid_size
        output[..., :2] *= stride
        output[..., 2] *= h.image_size

        return output
    
    def load_base_weights(self, filepath: str) -> None:
        blocks = self.blocks[1:]

        with open(filepath, 'rb') as file:
            # note: first 5 values are irrelevant
            np.fromfile(file, dtype=np.int32, count=5)

            weights = np.fromfile(file, dtype=np.float32)
            weights_idx = 0

            for idx, block in enumerate(blocks):
                layer_type: str = block['type']
                if layer_type != 'convolutional':
                    continue

                activation_type: str = block['activation']
                if activation_type == 'linear':
                    continue

                layer: torch.nn.Module = self.neural_network[idx]
                should_normalize = bool(block.get('batch_normalize'))

                conv: torch.nn.Conv2d = layer[0]

                if should_normalize:
                    bn: torch.nn.BatchNorm2d = layer[1]
                    number_of_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[weights_idx: weights_idx + number_of_bn_biases])
                    weights_idx += number_of_bn_biases

                    bn_weights = torch.from_numpy(weights[weights_idx: weights_idx + number_of_bn_biases])
                    weights_idx  += number_of_bn_biases

                    bn_running_mean = torch.from_numpy(weights[weights_idx: weights_idx + number_of_bn_biases])
                    weights_idx  += number_of_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[weights_idx: weights_idx + number_of_bn_biases])
                    weights_idx  += number_of_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn.bias.data.copy_(bn_biases)

                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn.weight.data.copy_(bn_weights)

                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn.running_mean.copy_(bn_running_mean)

                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.running_var.copy_(bn_running_var)
                else:
                    number_of_bn_biases = conv.bias.numel()
                
                    conv_biases = torch.from_numpy(weights[weights_idx: weights_idx + number_of_bn_biases])
                    weights_idx  += number_of_bn_biases
                
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                number_of_weights = conv.weight.numel()
                
                conv_weights = torch.from_numpy(weights[weights_idx: weights_idx + number_of_weights])
                weights_idx  += number_of_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def _read_config(self, filepath: str) -> list[dict]:
        with open(filepath) as file:
            lines = file.read().split('\n')
            lines = [line.strip() for line in lines if len(line) > 0 and not line.startswith('#')]

            blocks: list[dict] = []
            current_block = {}
            for line in lines:
                if line.startswith('['):
                    if (len(current_block) > 0):
                        blocks.append(current_block)

                    current_block = {}
                    current_block['type'] = line[1: -1].strip()
                else:
                    key, value = line.split('=')
                    current_block[key.strip()] = value.strip()

            blocks.append(current_block)
            return blocks

    def _create_modules(self, blocks: list[dict]) -> torch.nn.ModuleList:
        modules = torch.nn.ModuleList()

        channels: list[int] = []

        for idx, block in enumerate(blocks[1:]):
            if block['type'] == 'convolutional':
                self._add_conv_module(idx, block, modules, channels)
                continue

            if block['type'] == 'shortcut':
                self._add_shortcut_module(block, modules, channels)
                continue

            if block['type'] == 'route':
                self._add_route_module(block, modules, channels)
                continue

            if block['type'] == 'yolo':
                self._add_output_module(block, modules, channels)
                continue

            if block['type'] == 'upsample':
                self._add_upsample_module(block, modules, channels)
                continue

        return modules

    def _add_conv_module(self, idx: int, block: dict, modules: torch.nn.ModuleList, channels: list[int]) -> None:
        module = torch.nn.Sequential()

        should_normalize = bool(block.get('batch_normalize'))
        should_activate = block.get('activation') == 'leaky'
        should_pad = bool(block.get('pad'))

        out_channels = int(block['filters'])
        kernel_size = int(block['size'])
        stride = int(block['stride'])

        in_channels = channels[-1] if len(channels) > 0 else 3
        bias = not should_normalize
        padding = kernel_size // 2 if should_pad else 0

        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        module.add_module(f'conv_{idx}', conv)

        if should_normalize:
            batch_norm = torch.nn.BatchNorm2d(out_channels)
            module.add_module(f'batch_norm_{idx}', batch_norm)

        if should_activate:
            leaky = torch.nn.LeakyReLU(0.1)
            module.add_module('leaky_{idx}', leaky)

        modules.append(module)
        channels.append(out_channels)

    def _add_shortcut_module(self, block: dict, modules: torch.nn.ModuleList, channels: list[int]) -> None:
        from_idx = int(block['from'])

        shortcut = Shortcut(from_idx)

        modules.append(shortcut)
        channels.append(channels[-1])

    def _add_route_module(self, block: dict, modules: torch.nn.ModuleList, channels: list[int]) -> None:
        indices = str(block['layers']).split(',')
        from_idx = int(indices[0])
        to_idx = int(indices[1]) if len(indices) > 1 else 0

        out_channels = channels[from_idx] + channels[to_idx] if to_idx > 0 else channels[from_idx]

        route = Route(from_idx, to_idx)

        modules.append(route)
        channels.append(out_channels)

    def _add_output_module(self, _: dict, modules: torch.nn.ModuleList, channels: list[int]) -> None:
        output = Output()

        modules.append(output)
        channels.append(-1)

    def _add_upsample_module(self, block: dict, modules: torch.nn.ModuleList, channels: list[int]) -> None:
        stride = int(block['stride'])

        upsample = torch.nn.Upsample(scale_factor=stride)

        modules.append(upsample)
        channels.append(channels[-1])