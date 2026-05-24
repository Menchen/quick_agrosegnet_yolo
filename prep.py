import os
import cv2
import yaml
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DataPreparation:
    def __init__(
        self,
        dataset_path="Menchen/AgroSegNet",
        subset="default-tiny",
        output_dir="./yolo_dataset",
    ):
        self.dataset_path = dataset_path
        self.subset = subset
        self.output_dir = output_dir

    def mask_to_yolo_polygons(self, mask_img):
        """Extracts contours via OpenCV and normalizes to YOLO format."""
        _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = mask_img.shape
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                contour = contour.flatten().astype(float)
                contour[0::2] /= w
                contour[1::2] /= h
                poly_str = "0 " + " ".join(f"{coord:.6f}" for coord in contour)
                polygons.append(poly_str)
        return polygons

    def export_split(self, dataset_split, split_name, img_col, mask_col):
        img_dir = os.path.join(self.output_dir, "images", split_name)
        lbl_dir = os.path.join(self.output_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i, item in enumerate(tqdm(dataset_split, desc=f"Processing {split_name}")):
            # Fast numpy conversion (bypassing Pillow ops)
            img = np.array(item[img_col])
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            mask = np.array(item[mask_col])
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            img_path = os.path.join(img_dir, f"{split_name}_{i}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{split_name}_{i}.txt")

            cv2.imwrite(img_path, img)

            polygons = self.mask_to_yolo_polygons(mask)
            with open(lbl_path, "w") as f:
                f.write("\n".join(polygons))

    def run(self):
        print(f"Loading dataset: {self.dataset_path} ({self.subset})...")
        ds = load_dataset(self.dataset_path, self.subset)

        # Dynamically detect column names
        cols = ds["train"].column_names
        img_col = "image" if "image" in cols else cols[0]
        mask_col = (
            "label" if "label" in cols else ("mask" if "mask" in cols else cols[1])
        )

        # Generate train/val splits from original train
        train_val = ds["train"].train_test_split(test_size=0.2, seed=42)

        self.export_split(train_val["train"], "train", img_col, mask_col)
        self.export_split(train_val["test"], "val", img_col, mask_col)

        # Handle test split if available
        if "test" in ds:
            self.export_split(ds["test"], "test", img_col, mask_col)

        # Generate YAML
        yaml_data = {
            "path": os.path.abspath(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test" if "test" in ds else "images/val",
            "names": {0: "shadow"},
        }

        yaml_path = os.path.join(self.output_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        print(f"Data preparation complete. YAML saved at: {yaml_path}")


if __name__ == "__main__":
    prep = DataPreparation()
    prep.run()
