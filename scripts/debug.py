import sys
sys.path.append("scripts")
from deepfashion_dataset import DeepFashionDataset, eval_transform

IMG_ROOT       = r"C:\Users\ranal\ai-fashion-automation\data\img_highres"
CATEGORY_FILE  = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_category_img.txt"
PARTITION_FILE = r"C:\Users\ranal\ai-fashion-automation\data\Eval\list_eval_partition.txt"

train_ds = DeepFashionDataset(IMG_ROOT, CATEGORY_FILE, PARTITION_FILE, split='train')

# Pass train class map to val
val_ds = DeepFashionDataset(IMG_ROOT, CATEGORY_FILE, PARTITION_FILE, 
                             split='val',
                             class_map=train_ds.class_map)  # ← add this

print("Train classes:", train_ds.classes)
print("Val classes:  ", val_ds.classes)
print("Train num_classes:", train_ds.num_classes)
print("Val num_classes:  ", val_ds.num_classes)
print("Train class_map:", train_ds.class_map)
print("Val class_map:  ", val_ds.class_map)
print("Maps match:", train_ds.class_map == val_ds.class_map)
