from ultralytics import YOLO


class ShadowSegmentationTrainer:
    def __init__(
        self,
        model_architecture="yolov8n-seg.pt",
        data_yaml="./yolo_dataset/dataset.yaml",
    ):
        # Initializing from a pre-trained model ensures faster convergence
        self.model = YOLO(model_architecture)
        self.data_yaml = data_yaml

    def train(
        self, epochs=50, imgsz=640, batch=16, project="agro_shadow", name="yolo_seg_run"
    ):
        print(f"Starting training for {epochs} epochs...")
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device="cuda",
            patience=10,  # Early stopping
            save=True,
        )
        return results


if __name__ == "__main__":
    # supported model:
    # yolo26n-seg.pt yolo26s-seg.pt yolo26m-seg.pt yolo26l-seg.pt yolo26x-seg.pt
    # yolo26n-sem.pt yolo26s-sem.pt yolo26m-sem.pt yolo26l-sem.pt yolo26x-sem.pt
    trainer = ShadowSegmentationTrainer(model_architecture="yolo26l-sem.pt")
    # Tweak hyper-parameters here depending on your VRAM limits
    trainer.train(epochs=50, batch=16)
