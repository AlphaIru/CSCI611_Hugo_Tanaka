"""This will convert pre-existing labels into txt format"""

# Base code by ChatGPT
# Modified by Hugo Tanaka
# Professor Bo Shen
# 2026/3/13
# CSCI 611

from os import path, makedirs
import json
import shutil

BASE:str = "dataset"

IMG_DIR:str = path.join(BASE, "images")
ANN_DIR:str = path.join(BASE, "mtsd_v2_fully_annotated", "annotations")
SPLIT_DIR:str = path.join(BASE, "mtsd_v2_fully_annotated", "splits")

OUT:str = path.join(BASE, "yolo_dataset")

splits:list[str] = ["train", "val", "test"]


def conversion():
    """This will convert the MTSD file into txt format"""

    for split in splits:
        makedirs(path.join(OUT, "images", split), exist_ok=True)
        makedirs(path.join(OUT, "labels", split), exist_ok=True)

    for split in splits:
        split_file:str = path.join(SPLIT_DIR, f"{split}.txt")
        with open(
                split_file,
                mode="r",
                encoding="utf-8"
                ) as opened_file:
            img_id_list: list[str] = []
            for line in opened_file:
                line = line.strip()

                if line != "":
                    img_id_list.append(line)

        for img_id in img_id_list:
            img_name:str = img_id + ".jpg"
            json_name:str = img_id + ".json"

            img_path:str = path.join(IMG_DIR, img_name)
            ann_path:str = path.join(ANN_DIR, json_name)

            if not path.exists(img_path):
                out_lbl_missing = path.join(OUT, "labels", split, img_id + ".txt")
                with open(out_lbl_missing, "w", encoding="utf-8") as _:
                    pass
                continue

            out_img: str = path.join(OUT, "images", split, img_name)
            shutil.copy2(img_path, out_img)

            yolo_lines = []

            # if annotation exists, parse it; otherwise produce empty txt
            if path.exists(ann_path):
                try:
                    with open(ann_path, mode="r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[WARN] can't read {ann_path}: {e}")
                    data = {}

                width:float = data.get("width")
                height:float = data.get("height")
                objects:float = data.get("objects", [])

                if width and height:
                    for obj in objects:
                        props = obj.get("properties", {})
                        included = props.get("included", True)
                        ambiguous = props.get("ambiguous", False)

                        if not included or ambiguous:
                            continue

                        bbox = obj.get("bbox", {})
                        xmin:float = bbox.get("xmin")
                        ymin:float = bbox.get("ymin")
                        xmax:float = bbox.get("xmax")
                        ymax:float = bbox.get("ymax")

                        if None in [xmin, ymin, xmax, ymax]:
                            continue
                        if xmax <= xmin or ymax <= ymin:
                            continue

                        x_center:float = ((xmin + xmax) / 2) / width
                        y_center:float = ((ymin + ymax) / 2) / height
                        box_w:float = (xmax - xmin) / width
                        box_h:float = (ymax - ymin) / height

                        # one-class dataset: all valid traffic signs = class 0
                        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

            output_label: str = path.join(OUT, "labels", split, img_id + ".txt")
            with open(output_label, mode="w", encoding="utf-8") as opened_file:
                if yolo_lines:
                    opened_file.write("\n".join(yolo_lines))
                else:
                    # write empty file when no objects (YOLO accepts this)
                    opened_file.write("")

    print("Conversion complete.")

if __name__ == "__main__":
    conversion()

