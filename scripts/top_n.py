import os
import shutil

SOURCE_DIR = r"E:\deepfashion-ml\data\images_cleaned"
OUTPUT_DIR = r"E:\deepfashion-ml\data\images_top100"
TOP_N = 100

# Count images per class
class_counts = {}
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

# Sort and take top N
top_classes = sorted(class_counts, key=class_counts.get, reverse=True)[:TOP_N]

os.makedirs(OUTPUT_DIR, exist_ok=True)
for class_name in top_classes:
    src = os.path.join(SOURCE_DIR, class_name)
    dst = os.path.join(OUTPUT_DIR, class_name)
    shutil.copytree(src, dst)

print(f"Done. Kept top {TOP_N} classes.")
for c in top_classes[:10]:
    print(f"  {c}: {class_counts[c]} images")