"""
Modular YOLO Segmentation Training Script for Menchen/AgroSegNet
Uses OpenCV for image/mask processing and Ultralytics for training.
"""

import cv2
import yaml
import numpy as np
from pathlib import Path
from datasets import load_dataset
from ultralytics import YOLO


class AgroSegNetDatasetBuilder:
    def __init__(
        self,
        dataset_path: str = "Menchen/AgroSegNet",
        subset: str = "default-tiny",
        output_dir: str = "agrosegnet_yolo",
    ):
        self.dataset_path = dataset_path
        self.subset = subset
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        # YOLO requires 'val' rather than 'test' for the validation set directory structure
        self.split_map = {"train": "train", "test": "val", "validation": "val"}

    def _prepare_directories(self):
        """Creates the necessary folder structure for Ultralytics YOLO."""
        for split in set(self.split_map.values()):
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    def extract_polygons(self, mask: np.ndarray, width: int, height: int) -> list:
        """
        Extracts normalized polygons from a mask using OpenCV.
        Assumes 0 is the background.
        """
        polygons = []

        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        # Find all unique class values excluding background (0)
        classes = np.unique(mask)
        classes = classes[classes != 0]

        for cls_idx, cls_val in enumerate(sorted(classes)):
            # Create a binary mask for the specific class
            binary_mask = np.where(mask == cls_val, 255, 0).astype(np.uint8)

            # Extract boundaries using OpenCV
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Approximate the contour to reduce total polygon points (optimizes YOLO training)
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # YOLO requires a minimum of 3 points (to form a valid polygon surface)
                if len(approx) >= 3:
                    poly = []
                    for point in approx:
                        x, y = point[0]
                        poly.extend([x / width, y / height])  # Normalize coordinates

                    polygons.append((cls_idx, poly))

        return polygons

    def build(self) -> str:
        """Downloads the dataset, processes images/masks, and formats them for YOLO."""
        print(f"Downloading {self.dataset_path} ({self.subset})...")
        hf_dataset = load_dataset(self.dataset_path, self.subset)
        self._prepare_directories()

        for hf_split, split_data in hf_dataset.items():
            yolo_split = self.split_map.get(hf_split, hf_split)
            print(
                f"Processing split: {hf_split} -> {yolo_split} ({len(split_data)} images)"
            )

            for idx, item in enumerate(split_data):
                # Convert HF images to numpy arrays to strictly use OpenCV processing
                img_np = np.array(item["image"])
                mask_np = np.array(item["label"])

                height, width = img_np.shape[:2]

                # Convert RGB (HF default) to BGR for OpenCV
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Save Image
                img_name = f"{yolo_split}_{idx:05d}.jpg"
                cv2.imwrite(str(self.images_dir / yolo_split / img_name), img_np)

                # Extract and save normalized polygons
                polygons = self.extract_polygons(mask_np, width, height)
                label_name = f"{yolo_split}_{idx:05d}.txt"

                with open(self.labels_dir / yolo_split / label_name, "w") as f:
                    for cls_id, poly in polygons:
                        poly_str = " ".join([f"{val:.6f}" for val in poly])
                        f.write(f"{cls_id} {poly_str}\n")

        return self._create_yaml()

    def _create_yaml(self) -> str:
        """Generates the data.yaml file mapping required by Ultralytics."""
        yaml_path = self.output_dir / "data.yaml"
        yaml_content = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "shadow"
            },  # Adjust class labeling if multi-class (e.g., sunlight vs shadow)
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print(f"Dataset configuration saved to {yaml_path}")
        return str(yaml_path)


class SegmentationTrainer:
    def __init__(self, model_version: str = "yolo11n-seg.pt"):
        """Initializes the Ultralytics segmentation model (defaults to YOLO11 nano)."""
        self.model = YOLO(model_version)

    def train(
        self, data_yaml: str, epochs: int = 50, imgsz: int = 640, batch: int = 16
    ):
        """Starts the training loop."""
        print(f"Starting training for {epochs} epochs...")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device="auto",  # Automatically detects and selects CUDA if available
            patience=10,  # Early stopping patience
            project="agrosegnet_training",
            name="shadow_seg_model",
        )
        return results


if __name__ == "__main__":
    # 1. Download and compile the Hugging Face dataset into YOLO polygon format
    builder = AgroSegNetDatasetBuilder(subset="default-tiny")
    yaml_config_path = builder.build()

    # 2. Spin up the model and start training
    trainer = SegmentationTrainer(model_version="yolo11n-seg.pt")
    trainer.train(data_yaml=yaml_config_path, epochs=30, imgsz=640)
