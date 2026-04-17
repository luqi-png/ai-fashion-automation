import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

import sys
sys.path.append(os.path.dirname(__file__))
from deepfashion_dataset import DeepFashionDataset, eval_transform

# ----------------------------
# Paths
# ----------------------------
IMG_ROOT       = r"C:\Users\ranal\ai-fashion-automation\data\img_highres"
CATEGORY_FILE  = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_category_img.txt"
PARTITION_FILE = r"C:\Users\ranal\ai-fashion-automation\data\Eval\list_eval_partition.txt"
MODEL_PATH     = r"C:\Users\ranal\ai-fashion-automation\models\efficientnet_b0_best.pth"

BATCH_SIZE = 32
MIN_IMAGES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Load train dataset first to get class map
# ----------------------------
train_dataset = DeepFashionDataset(
    IMG_ROOT, CATEGORY_FILE, PARTITION_FILE,
    split='train', transform=eval_transform, min_images=MIN_IMAGES
)

# ----------------------------
# Load test dataset using train class map
# ----------------------------
test_dataset = DeepFashionDataset(
    IMG_ROOT, CATEGORY_FILE, PARTITION_FILE,
    split='test', transform=eval_transform, min_images=MIN_IMAGES,
    class_map=train_dataset.class_map
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0, pin_memory=True)

print(f"Number of classes: {test_dataset.num_classes}")
print(f"Classes: {test_dataset.classes}")

# ----------------------------
# Load EfficientNet-B0 model
# ----------------------------
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, test_dataset.num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.to(device)
model.eval()
print("Model loaded successfully")

# ----------------------------
# Evaluate
# ----------------------------
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {correct/total:.4f} ({correct}/{total})")