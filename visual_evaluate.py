import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from datasets import load_dataset
from tqdm import tqdm


class HeadlessVisualEvaluator:
    def __init__(
        self,
        model_path,
        dataset_path="Menchen/AgroSegNet",
        subset="default-tiny",
        output_dir="./visual_evals",
    ):
        self.model = YOLO(model_path)
        self.dataset_path = dataset_path
        self.subset = subset
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def create_colored_overlay(self, img, binary_mask, color_bgr):
        """
        Blends a colored tint over the original image where the binary_mask is true.
        """
        overlay = img.copy()
        # Apply color where mask is > 0
        overlay[binary_mask > 0] = color_bgr
        # Blend with original image (alpha=0.4 for overlay, beta=0.6 for original)
        return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    def add_label(self, img, text):
        """Adds a clear background text label to the top-left of the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Draw black background rectangle for text visibility
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        cv2.rectangle(img, (0, 0), (text_width + 20, text_height + 20), (0, 0, 0), -1)
        # Draw white text
        cv2.putText(
            img,
            text,
            (10, text_height + 10),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        return img

    def run(self, num_samples=15):
        print(f"Loading dataset: {self.dataset_path} ({self.subset})...")
        ds = load_dataset(self.dataset_path, self.subset)
        cols = ds["train"].column_names
        img_col = "image" if "image" in cols else cols[0]
        mask_col = (
            "label" if "label" in cols else ("mask" if "mask" in cols else cols[1])
        )

        # Use test split if available, otherwise carve out a random validation split
        test_ds = (
            ds["test"]
            if "test" in ds
            else ds["train"].train_test_split(test_size=0.2, seed=42)["test"]
        )

        # Select random indices to evaluate
        total_items = len(test_ds)
        sample_indices = random.sample(
            range(total_items), min(num_samples, total_items)
        )

        print(
            f"Generating {len(sample_indices)} visual comparisons in '{self.output_dir}'..."
        )

        for idx, i in enumerate(tqdm(sample_indices, desc="Rendering Panels")):
            item = test_ds[i]

            # 1. Process Input Image (Safety checks for 1-channel / RGBA)
            img = np.array(item[img_col])
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img

            # 2. Process Ground Truth Mask
            gt_mask = np.array(item[mask_col])
            if len(gt_mask.shape) == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)
            _, gt_mask_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

            # 3. Predict with YOLO
            results = self.model.predict(img_bgr, verbose=False, imgsz=640)

            h, w = img_bgr.shape[:2]
            pred_mask_bin = np.zeros((h, w), dtype=np.uint8)

            # Reconstruct YOLO polygon prediction into a blank OpenCV mask
            if results[0].masks is not None:
                for polygon in results[0].masks.xy:
                    if len(polygon) > 0:
                        poly = np.array(polygon, dtype=np.int32)
                        cv2.fillPoly(pred_mask_bin, [poly], 255)

            # 4. Generate Visual Overlays
            # Green tint for Ground Truth, Red tint for Prediction
            gt_overlay = self.create_colored_overlay(
                img_bgr, gt_mask_bin, color_bgr=(0, 255, 0)
            )
            pred_overlay = self.create_colored_overlay(
                img_bgr, pred_mask_bin, color_bgr=(0, 0, 255)
            )

            # Add Text Labels
            img_bgr_labeled = self.add_label(img_bgr.copy(), "Original Image")
            gt_overlay_labeled = self.add_label(gt_overlay, "Ground Truth (Green)")
            pred_overlay_labeled = self.add_label(pred_overlay, "Prediction (Red)")

            # 5. Stitch horizontally into a single 3-panel image
            combined_panel = cv2.hconcat(
                [img_bgr_labeled, gt_overlay_labeled, pred_overlay_labeled]
            )

            # Ensure image fits standard displays (resize down if horizontal concat is massive)
            max_width = 1920
            if combined_panel.shape[1] > max_width:
                scale = max_width / combined_panel.shape[1]
                new_dim = (max_width, int(combined_panel.shape[0] * scale))
                combined_panel = cv2.resize(
                    combined_panel, new_dim, interpolation=cv2.INTER_AREA
                )

            # 6. Save to disk securely (headless-safe)
            output_filepath = os.path.join(self.output_dir, f"sample_{idx}_img_{i}.jpg")
            cv2.imwrite(output_filepath, combined_panel)

        print(
            f"\nDone! Download the '{self.output_dir}' folder to your local machine to view the results."
        )


if __name__ == "__main__":
    # Point this to your best weights from the previous training run
    best_weights = "./runs/segment/agro_shadow/yolo_seg_run/weights/best.pt"

    if os.path.exists(best_weights):
        visualizer = HeadlessVisualEvaluator(model_path=best_weights)
        # Change num_samples to generate as many test cases as you want to review
        visualizer.run(num_samples=15)
    else:
        print(
            f"Error: Model not found at {best_weights}. Please verify your training output path."
        )
