import os
import numpy as np
from sklearn.cluster import KMeans

def convert_labels(input_dir: str, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename in os.listdir(input_dir):
        output_filepath = os.path.join(output_dir, filename)
        output_file_lines: list[str] = []

        filepath = os.path.join(input_dir, filename)
        with open(filepath, mode='r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.strip().split(' ')
                x, y, w, h = float(values[1]), float(values[2]), float(values[3]), float(values[4])
                d = max(w, h)
                r = d / 2

                # note: disregard "edge cases"
                if x + r >= 1 or y + r >= 1 or x - r <= 0 or y - r <= 0:
                    continue

                output_file_lines.append(' '.join([str(value) for value in [0, x, y, d, d]]))

        with open(output_filepath, mode='w') as file:
            file.write('\n'.join(output_file_lines))

def calculate_anchors(input_dir: str) -> None:
    diameters: list[float] = []

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, mode='r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.strip().split(' ')
                diameters.append(float(values[3]))

    diameters = np.array(diameters).reshape(-1, 1)
    anchors = kmeans_anchors(diameters, k=9)
    for anchor in anchors:
        print(f'{anchor:4f}')

def kmeans_anchors(diameters: np.ndarray, k: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(diameters)
    anchor_sizes = kmeans.cluster_centers_
    anchor_sizes = sorted(anchor_sizes.flatten())
    return anchor_sizes
