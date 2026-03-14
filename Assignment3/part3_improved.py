"""This will train a YOLO model for small object detection."""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def main():
    """This will create a new_model"""

    model.train(
        data="dataset/yolo_dataset/dataset.yaml",
        epochs=8,
        imgsz=1280,             # increased the resolution from 640 to 1280
        batch=8,
        scale=0.5,              # Simulates zoom-out
        degrees=5,              # Adds rotation to the image
        perspective=0.0005,     # Adds distortion
        mixup=0.1,               # Combines images together.
        name="traffic_sign_aug_1280"
    )


if __name__ == "__main__":
    main()

