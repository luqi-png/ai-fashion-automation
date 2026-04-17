import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

import sys
sys.path.append(os.path.dirname(__file__))
from deepfashion_dataset import DeepFashionDataset, train_transform, eval_transform

# ----------------------------
# Paths
# ----------------------------
IMG_ROOT       = r"C:\Users\ranal\ai-fashion-automation\data\img_highres"
CATEGORY_FILE  = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_category_img.txt"
PARTITION_FILE = r"C:\Users\ranal\ai-fashion-automation\data\Eval\list_eval_partition.txt"

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE    = 32
FREEZE_EPOCHS = 5    # Phase 1: train classifier only
TOTAL_EPOCHS  = 30   # Phase 2: fine-tune whole network
LR_FROZEN     = 1e-3 # Higher LR when backbone is frozen
LR_UNFROZEN   = 1e-4 # Lower LR when backbone is unfrozen
MIN_IMAGES    = 100

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Datasets
# ----------------------------
train_dataset = DeepFashionDataset(
    IMG_ROOT, CATEGORY_FILE, PARTITION_FILE,
    split='train', transform=train_transform, min_images=MIN_IMAGES
)
val_dataset = DeepFashionDataset(
    IMG_ROOT, CATEGORY_FILE, PARTITION_FILE,
    split='val', transform=eval_transform, min_images=MIN_IMAGES,
    class_map=train_dataset.class_map
)

print(f"Classes: {train_dataset.classes}")
print(f"Number of classes: {train_dataset.num_classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

# ----------------------------
# Class weights (square root smoothing)
# ----------------------------
print("Calculating class weights...")
class_counts = torch.zeros(train_dataset.num_classes)
for _, label in train_dataset.samples:
    class_counts[label] += 1

# Square root smoothing — less aggressive than raw inverse frequency
class_weights = 1.0 / torch.sqrt(class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * train_dataset.num_classes
class_weights = class_weights.to(device)
print(f"Class weights — Min: {class_weights.min():.4f} Max: {class_weights.max():.4f}")

# ----------------------------
# Model — EfficientNet-B0
# ----------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, train_dataset.num_classes)
)
model.to(device)
print("Model: EfficientNet-B0 with pretrained ImageNet weights")

# ----------------------------
# Loss function
# ----------------------------
#criterion = nn.CrossEntropyLoss(weight=class_weights)
# Replace weighted loss with standard loss
criterion = nn.CrossEntropyLoss()

# ----------------------------
# Phase 1: Freeze backbone
# Train only the classifier for FREEZE_EPOCHS
# ----------------------------
print(f"\n--- Phase 1: Frozen backbone ({FREEZE_EPOCHS} epochs, LR={LR_FROZEN}) ---")
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_FROZEN, weight_decay=1e-4
)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

best_val_acc = 0.0

for epoch in range(FREEZE_EPOCHS):
    # --- Train ---
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    train_acc = train_correct / train_total

    # --- Validate ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/efficientnet_b0_best.pth")

    print(f"[Phase 1] Epoch [{epoch+1:02d}/{FREEZE_EPOCHS}] "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Best: {best_val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

# ----------------------------
# Phase 2: Unfreeze backbone
# Fine-tune whole network
# ----------------------------
remaining_epochs = TOTAL_EPOCHS - FREEZE_EPOCHS
print(f"\n--- Phase 2: Unfrozen backbone ({remaining_epochs} epochs, LR={LR_UNFROZEN}) ---")
for param in model.features.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LR_UNFROZEN, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

for epoch in range(remaining_epochs):
    # --- Train ---
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    train_acc = train_correct / train_total

    # --- Validate ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/efficientnet_b0_best.pth")

    print(f"[Phase 2] Epoch [{epoch+1:02d}/{remaining_epochs}] "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Best: {best_val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print("Model saved to models/efficientnet_b0_best.pth")