import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm




PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(PROJECT_DIR, "MAR20")
IMAGE_DIR = os.path.join(BASE_DIR, "JPEGImages")
XML_DIR = os.path.join(BASE_DIR, "Annotations", "Horizontal Bounding Boxes")
SPLIT_DIR = os.path.join(BASE_DIR, "ImageSets", "Main")


OUTPUT_DIR = os.path.join(BASE_DIR, "Classification_Dataset")
MARGIN_RATIO = 0.10
CLEAN_OUTPUT_SPLITS = True
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

def create_padded_crop(img, box, margin_ratio=MARGIN_RATIO):
    xmin, ymin, xmax, ymax = box
    width, height = xmax - xmin, ymax - ymin


    margin_x, margin_y = int(width * margin_ratio), int(height * margin_ratio)


    new_xmin = max(0, xmin - margin_x)
    new_ymin = max(0, ymin - margin_y)
    new_xmax = min(img.width, xmax + margin_x)
    new_ymax = min(img.height, ymax + margin_y)


    cropped_img = img.crop((new_xmin, new_ymin, new_xmax, new_ymax))


    max_side = max(cropped_img.size)
    padded_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))


    offset = ((max_side - cropped_img.width) // 2, (max_side - cropped_img.height) // 2)
    padded_img.paste(cropped_img, offset)


    return padded_img.resize((448, 448), Image.Resampling.BICUBIC)

def parse_box_from_object(obj):
    bndbox = obj.find('bndbox')
    if bndbox is None:
        return None
    return [
        int(float(bndbox.find('xmin').text)),
        int(float(bndbox.find('ymin').text)),
        int(float(bndbox.find('xmax').text)),
        int(float(bndbox.find('ymax').text))
    ]

def process_ids(filenames, target_folder, margin_ratio=MARGIN_RATIO):
    for filename in tqdm(filenames, desc=f"Processing {target_folder}"):
        img_path = os.path.join(IMAGE_DIR, f"{filename}.jpg")
        xml_path = os.path.join(XML_DIR, f"{filename}.xml")

        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue

        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj_idx, obj in enumerate(root.findall('object')):
            cls_name = obj.find('name').text
            box = parse_box_from_object(obj)
            if box is None:
                continue


            cls_dir = os.path.join(OUTPUT_DIR, target_folder, cls_name)
            os.makedirs(cls_dir, exist_ok=True)


            result_img = create_padded_crop(img, box, margin_ratio=margin_ratio)
            result_img.save(os.path.join(cls_dir, f"{filename}_{obj_idx}.jpg"))

def read_split_ids(split_filename):
    split_path = os.path.join(SPLIT_DIR, split_filename)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, 'r') as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]

    unique_ids = list(dict.fromkeys(ids))
    if len(unique_ids) != len(ids):
        print(f"Warning: duplicated IDs in {split_filename} -> deduplicated {len(ids)} to {len(unique_ids)}")
    return unique_ids

def make_train_val_split_from_train_txt(train_ids, train_split_ratio=TRAIN_SPLIT_RATIO, random_seed=RANDOM_SEED):
    if len(train_ids) < 2:
        raise ValueError("Need at least 2 samples in train.txt for train/val split.")

    shuffled = train_ids.copy()
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_split_ratio)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    return shuffled[:split_idx], shuffled[split_idx:]

def save_split_ids(train_ids, val_ids, test_ids):
    split_output_dir = os.path.join(OUTPUT_DIR, "splits")
    os.makedirs(split_output_dir, exist_ok=True)

    train_out = os.path.join(split_output_dir, "train_from_train_txt.txt")
    val_out = os.path.join(split_output_dir, "val_from_train_txt.txt")
    test_out = os.path.join(split_output_dir, "test_from_test_txt.txt")

    with open(train_out, "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(val_out, "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids) + "\n")
    with open(test_out, "w", encoding="utf-8") as f:
        f.write("\n".join(test_ids) + "\n")

    print(f"Saved split IDs: {train_out}, {val_out}, {test_out}")

def validate_splits(train_ids, val_ids, test_ids):
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if overlap_train_val:
        sample = sorted(list(overlap_train_val))[:10]
        raise ValueError(f"Data leakage detected (train-val overlap). Sample: {sample}")
    if overlap_train_test:
        sample = sorted(list(overlap_train_test))[:10]
        raise ValueError(f"Data leakage detected (train-test overlap). Sample: {sample}")
    if overlap_val_test:
        sample = sorted(list(overlap_val_test))[:10]
        raise ValueError(f"Data leakage detected (val-test overlap). Sample: {sample}")

    print(
        "Split validation passed - "
        f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}, overlaps: 0"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build MAR20 classification crops from detection images and HBB XML annotations."
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=TRAIN_SPLIT_RATIO,
        help=f"Ratio of train.txt image IDs used for train split. Default: {TRAIN_SPLIT_RATIO}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for train/val split. Default: {RANDOM_SEED}",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=MARGIN_RATIO,
        help=f"Extra crop margin around bounding boxes. Default: {MARGIN_RATIO}",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing train/val/test output folders before writing crops.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if not 0 < args.train_split_ratio < 1:
        raise ValueError("--train-split-ratio must be between 0 and 1.")
    if args.margin_ratio < 0:
        raise ValueError("--margin-ratio must be non-negative.")

    source_train_ids = read_split_ids("train.txt")
    test_ids = read_split_ids("test.txt")
    train_ids, val_ids = make_train_val_split_from_train_txt(
        source_train_ids,
        train_split_ratio=args.train_split_ratio,
        random_seed=args.seed,
    )
    print(
        f"Split from train.txt with ratio {args.train_split_ratio:.2f}: "
        f"train={len(train_ids)}, val={len(val_ids)}, test(from test.txt)={len(test_ids)}"
    )
    validate_splits(train_ids, val_ids, test_ids)
    save_split_ids(train_ids, val_ids, test_ids)

    clean_output_splits = CLEAN_OUTPUT_SPLITS and not args.no_clean
    if clean_output_splits:
        for split_name in ("train", "val", "test"):
            split_dir = os.path.join(OUTPUT_DIR, split_name)
            if os.path.exists(split_dir):
                print(f"Removing existing split folder: {split_dir}")
                shutil.rmtree(split_dir)
    else:
        print("Keeping existing split folders because --no-clean was provided.")


    process_ids(train_ids, "train", margin_ratio=args.margin_ratio)
    process_ids(val_ids, "val", margin_ratio=args.margin_ratio)
    process_ids(test_ids, "test", margin_ratio=args.margin_ratio)
    print("Dataset preparation completed!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
