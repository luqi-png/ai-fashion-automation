import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models

from transforms import train_transform, eval_transform
from torch.utils.data import DataLoader, Subset



# -------------------
# Paths
# -------------------
TRAIN_DIR = r"E:/deepfashion-ml/data/train"
VAL_DIR   = r"E:/deepfashion-ml/data/val"

# -------------------
# Hyperparameters
# -------------------
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------
# Datasets
# -------------------

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = ImageFolder(VAL_DIR,   transform=eval_transform)

# Temporary: limit size for testing
train_dataset = Subset(train_dataset, range(500))
val_dataset   = Subset(val_dataset,   range(100))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.dataset.classes)
print("Number of classes:", num_classes)

# -------------------
# Model
# -------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# -------------------
# Loss & Optimizer
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------
# Training loop
# -------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # -------------------
    # Validation
    # -------------------
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

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Acc: {val_acc:.4f}")

print("Training complete.")