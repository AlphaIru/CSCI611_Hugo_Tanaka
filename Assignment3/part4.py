"""Run inference using trained YOLO model."""

from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/traffic_sign_aug_1280/weights/best.pt")

    model(
        "dataset/yolo_dataset/images/test",
        save=True,
        conf=0.5
    )


if __name__ == "__main__":
    main()
