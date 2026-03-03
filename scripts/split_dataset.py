import os
import random
import shutil

SOURCE_DIR = "E:\\deepfashion-ml\\data\\images_top100"
OUTPUT_DIR = "E:\\deepfashion-ml\\data"
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

random.seed(42)

for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(SPLITS["train"] * n)
    val_end = train_end + int(SPLITS["val"] * n)

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in split_map.items():
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)

print(" Dataset split completed.")