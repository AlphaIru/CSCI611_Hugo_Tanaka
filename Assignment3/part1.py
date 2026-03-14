"""This is a Assignment 3 code."""

# Written by Hugo Tanaka
# CSCI 611
# Professor Bo Shen
# 2026/3/13

from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def main():
    """This will run the main code."""

    # This is for test purpose
    # print("Hello World!")

    # Modded to save the output image
    results = model("sample_image.jpg", save=True)
    results[0].show()

    return


if __name__ == "__main__":
    main()
