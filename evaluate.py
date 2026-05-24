import os
import cv2
import numpy as np
from ultralytics import YOLO
from datasets import load_dataset
from tqdm import tqdm


class BEREvaluator:
    def __init__(
        self, model_path, dataset_path="Menchen/AgroSegNet", subset="default-tiny"
    ):
        self.model = YOLO(model_path)
        self.dataset_path = dataset_path
        self.subset = subset

    def evaluate(self):
        ds = load_dataset(self.dataset_path, self.subset)
        cols = ds["train"].column_names
        img_col = "image" if "image" in cols else cols[0]
        mask_col = (
            "label" if "label" in cols else ("mask" if "mask" in cols else cols[1])
        )

        # Pull test set, fallback to our val split logic if test doesn't exist
        test_ds = (
            ds["test"]
            if "test" in ds
            else ds["train"].train_test_split(test_size=0.2, seed=42)["test"]
        )

        # Global accumulators for dataset-wide BER
        total_TP = total_TN = total_FP = total_FN = 0

        for item in tqdm(test_ds, desc="Evaluating BER on Test Set"):
            img = np.array(item[img_col])

            # --- NEW: Force conversion to 3-channel BGR for YOLO ---
            if len(img.shape) == 2:
                # Grayscale (H, W)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                # Grayscale with channel dim (H, W, 1)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # Standard RGB (H, W, 3)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                # RGBA (H, W, 4)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img
            # -------------------------------------------------------

            gt_mask = np.array(item[mask_col])

            gt_mask = np.array(item[mask_col])
            if len(gt_mask.shape) == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)
            gt_bool = gt_mask > 0

            # Predict and parse normalized masks
            results = self.model.predict(img_bgr, verbose=False, imgsz=640)

            h, w = gt_mask.shape
            pred_mask = np.zeros((h, w), dtype=np.uint8)

            # Reconstruct prediction mask using OpenCV fillPoly on native image dims
            if results[0].masks is not None:
                for polygon in results[0].masks.xy:
                    if len(polygon) > 0:
                        poly = np.array(polygon, dtype=np.int32)
                        cv2.fillPoly(pred_mask, [poly], 255)

            pred_bool = pred_mask > 0

            # Update global confusion matrix
            total_TP += np.sum(gt_bool & pred_bool)
            total_TN += np.sum(~gt_bool & ~pred_bool)
            total_FP += np.sum(~gt_bool & pred_bool)
            total_FN += np.sum(gt_bool & ~pred_bool)

        # Calculate final dataset BER
        shadow_acc = (
            total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 1.0
        )
        non_shadow_acc = (
            total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 1.0
        )
        ber = 1.0 - 0.5 * (shadow_acc + non_shadow_acc)

        print("\n--- Evaluation Results ---")
        print(f"Shadow Accuracy (Sensitivity): {shadow_acc * 100:.2f}%")
        print(f"Non-Shadow Accuracy (Specificity): {non_shadow_acc * 100:.2f}%")
        print(f"Balanced Error Rate (BER): {ber * 100:.2f}% (Lower is better)")
        return ber


if __name__ == "__main__":
    # Point this to the dynamically generated best weights
    weights_path = "./runs/segment/agro_shadow/yolo_seg_run-2/weights/best.pt"
    if os.path.exists(weights_path):
        evaluator = BEREvaluator(model_path=weights_path)
        evaluator.evaluate()
    else:
        print(f"Model weights not found at {weights_path}. Run training first.")
