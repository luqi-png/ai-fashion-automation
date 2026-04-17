# Milestone 2 — Progress Report

## ✅ Completed Tasks

### Dataset & Preprocessing
- Set up project repository and folder structure on GitHub
- Collected and organized DeepFashion Category and Attribute Prediction Benchmark (289,222 images across 50 classes)
- Switched from folder-based label structure to official annotation files (`list_category_img.txt`, `list_eval_partition.txt`)
- Used official train/val/test partition (209,222 / 40,000 / 40,000 images)
- Filtered dataset to 35 classes by removing classes with fewer than 100 images per training split
- Implemented custom `PadToSquare` transform — pads images to square with black borders, preserving aspect ratio
- Implemented normalization using ImageNet mean and standard deviation
- Applied data augmentation for training set: random horizontal flip, color jitter, random rotation (±15°)
- Fixed critical label mismatch bug — class map is now built once from training set and passed explicitly to val/test splits

### Model — Component 1: Image Classification
- Built baseline CNN using pretrained ResNet-18 with fine-tuned classification head (70% val accuracy)
- Upgraded to EfficientNet-B0 following supervisor feedback — better accuracy-to-compute ratio
- Implemented two-phase freeze/unfreeze training strategy:
  - Phase 1 (5 epochs): backbone frozen, classifier only trained at LR=1e-3
  - Phase 2 (25 epochs): full network fine-tuned at LR=1e-4
- Replaced fixed StepLR scheduler with ReduceLROnPlateau (patience=3, factor=0.5)
- Added Dropout (p=0.3) before classifier to reduce overfitting
- Experimented with inverse square root class weighting to address 481x class imbalance — reverted to standard CrossEntropyLoss as weights slightly reduced accuracy
- **Final test accuracy: 70.68% (28,190 / 39,883 images)**

### Component 2: Image Tagging
- Loaded 1,000 attribute annotations from official `list_attr_img.txt`
- Built `image_tagger.py` — classifies an image and returns positive attribute tags
- Tags cover texture, fabric, shape, part, and style attribute types
- Verified on sample test images — produces meaningful tags alongside category predictions

### Component 3: NLP Description Generation
- Built `description_generator.py` — template-based natural language generation
- Detects fabric, texture, silhouette, and occasion keywords from tags
- Varies language based on model confidence score
- Produces human-readable descriptions from category + tags alone

### Full Pipeline
- Built `main.py` connecting all three components end-to-end
- Single image input → category + confidence + tags + description output
- Verified pipeline runs correctly on test set images

---

## 📦 Deliverables

| File | Description |
|------|-------------|
| `scripts/deepfashion_dataset.py` | Custom PyTorch Dataset using official DeepFashion annotations |
| `scripts/train.py` | Full training pipeline with freeze/unfreeze, ReduceLROnPlateau, dropout |
| `scripts/evaluate.py` | Test set evaluation script |
| `scripts/image_tagger.py` | Component 2 — image classification + attribute tagging |
| `scripts/description_generator.py` | Component 3 — NLP description generation |
| `scripts/main.py` | Full end-to-end pipeline connecting all three components |
| `scripts/debug.py` | Dataset diagnostic and class map verification tool |
| `models/efficientnet_b0_best.pth` | Best trained model checkpoint (Phase 2, Epoch 12) |
| `docs/AI_Fashion_Thesis_Progress_Report.docx` | Comprehensive progress report for supervisor |

---

## 📊 Experimental Results Summary

| Run | Model | Setup | Best Val Acc | Notes |
|-----|-------|-------|-------------|-------|
| 1 | ResNet-18 | Folder-based, Subset(10k) | 90% | Misleading — incorrect dataset setup |
| 2 | ResNet-18 | Full dataset, wrong class map | 1% | Critical label mismatch bug |
| 3 | ResNet-18 | Full dataset, fixed class map | 70% | First honest baseline |
| 4 | EfficientNet-B0 | Weighted loss (sqrt) | 69.6% | Weights too aggressive |
| 5 | EfficientNet-B0 | Standard loss, freeze/unfreeze | **71.3%** | Best — selected as final model |

**Final Test Accuracy: 70.68%**

---

## 🔜 Next Steps

- Build Flask frontend — image upload → category + tags + description displayed in browser
- Write thesis methodology chapter
- Prepare for thesis defense

---

## 📝 TODO

- [ ] Flask web frontend
- [ ] Thesis methodology write-up
- [ ] Defense preparation
