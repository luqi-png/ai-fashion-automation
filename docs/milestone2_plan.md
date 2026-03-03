# Milestone 2 

## Completed Tasks:

- Collected and organized DeepFashion dataset (50,000+ images, folder-per-class structure)
- Filtered dataset by removing classes with fewer than 20 images (399 classes removed)
- Reduced to top 100 classes by image count for manageable training scope
- Implemented image preprocessing pipeline (aspect ratio-preserving resize, padding to square, normalization)
- Split dataset into train/val/test sets (70/15/15 ratio)
- Implemented custom PyTorch Dataset and DataLoader using ImageFolder
- Defined data augmentation transforms for training set
- Built baseline CNN model using pretrained ResNet-18 with fine-tuned classification head
- Conducted preliminary training run — model converges successfully (Val Acc: 77% on subset)
- Verified end-to-end pipeline: image loading → preprocessing → model → prediction
## TODO
- Implement the core algorithm or model
- Integrate backend components and ensure basic functionality
- Start dataset collection and preprocessing
- Conduct preliminary tests and document architecture
