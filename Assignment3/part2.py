"""This will run the inference of given MTSD test image sets."""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    """The main test function."""
    
    model(
        "dataset/yolo_dataset/images/test",
        save=True,
        conf=0.5
    )


if __name__ == "__main__":
    main()

