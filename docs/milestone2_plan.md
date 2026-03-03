# Milestone 2 

## Completed Tasks

- Set up project repository and folder structure on GitHub
- Collected and organized DeepFashion dataset (50,000+ images, folder-per-class structure)
- Filtered dataset by removing classes with fewer than 20 images (399 classes removed)
- Reduced to top 100 classes by image count for a manageable training scope
- Implemented image preprocessing pipeline (aspect ratio-preserving resize, padding to square, normalization)
- Split dataset into train/val/test sets (70/15/15 ratio)
- Implemented custom PyTorch Dataset and DataLoader using `ImageFolder`
- Defined data augmentation transforms for training set (random horizontal flip)
- Built baseline CNN model using pretrained ResNet-18 with fine-tuned classification head
- Conducted preliminary training run — model converges successfully (Val Acc: 77% on subset)
- Verified end-to-end pipeline: image loading → preprocessing → model → prediction

## Deliverables

| File | Description |
|------|-------------|
| `scripts/transforms.py` | Preprocessing and augmentation pipeline |
| `scripts/split_dataset.py` | Reproducible dataset splitting script (70/15/15) |
| `scripts/train.py` | Full training loop with validation accuracy reporting |
| `data/train` | Training split (70% of images per class) |
| `data/val` | Validation split (15% of images per class) |
| `data/test` | Test split (15% of images per class) |
| `models/resnet18_deepfashion.pth` | Baseline ResNet-18 model weights *(saved after full training run)* |

## 🔜 Next Steps

- Run full training on complete dataset (10 epochs)
- Evaluate model on test set (accuracy, top-5 accuracy)
- Document results and model architecture in thesis
## TODO
- Implement the core algorithm or model
- Integrate backend components and ensure basic functionality
- Start dataset collection and preprocessing
- Conduct preliminary tests and document architecture
