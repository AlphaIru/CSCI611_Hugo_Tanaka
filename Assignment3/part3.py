"""This will train a YOLO model for small object detection."""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def main():
    """This will create a new_model"""

    model.train(
        data="dataset/yolo_dataset/dataset.yaml",
        epochs=8,
        imgsz=640,
        batch=16,
        name="traffic_sign_640"
    )


if __name__ == "__main__":
    main()

