import os
import shutil

SOURCE_DIR = r"E:\deepfashion-ml\img\img"
CLEANED_DIR = r"E:\deepfashion-ml\data\images_cleaned"
MIN_IMAGES = 20

os.makedirs(CLEANED_DIR, exist_ok=True)

removed = 0
kept = 0

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path)]
    if len(images) >= MIN_IMAGES:
        shutil.copytree(class_path, os.path.join(CLEANED_DIR, class_name))
        kept += 1
    else:
        removed += 1

print(f"Kept: {kept} classes, Removed: {removed} classes")