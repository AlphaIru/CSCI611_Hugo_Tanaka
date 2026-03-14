How to run the files.

## Setup

Beforehand, please make a Python environment, with all of the dependencies, so it can run all python programs.

Also, change the current directory to Assignment3.

Also this requires [Map Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign) to work.

Please download the following:
- mtsd_fully_annotated_annotation.zip
- mtsd_fully_annotated_images.test.zip
- mtsd_fully_annotated_images.train.0.zip
- mtsd_fully_annotated_images.train.1.zip
- mtsd_fully_annotated_images.train.2.zip
- mtsd_fully_annotated_images.val.zip

Place these files inside the `dataset/` folder.
Make sure to unzip/extract all files before running.

For part1:
Please run `python3 ./part1.py`.

For part2:
After extracting the zip files. Please run: `python3 ./convert_mtsd_to_yolo.py` and then run `python3 ./part2.py`.
