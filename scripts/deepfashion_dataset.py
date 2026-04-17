import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps



Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check

# ----------------------------
# Category names (50 classes)
# ----------------------------
CATEGORY_NAMES = [
    "Anorak", "Blazer", "Blouse", "Bomber", "Button-Down",
    "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
    "Jacket", "Jersey", "Parka", "Peacoat", "Poncho",
    "Sweater", "Tank", "Tee", "Top", "Turtleneck",
    "Capris", "Chinos", "Culottes", "Cutoffs", "Gauchos",
    "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
    "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts",
    "Trunks", "Caftan", "Cape", "Coat", "Coverup",
    "Dress", "Jumpsuit", "Kaftan", "Kimono", "Nightdress",
    "Onesie", "Robe", "Romper", "Shirtdress", "Sundress"
]

# Minimum images required to include a class
MIN_IMAGES = 100


class PadToSquare:
    """Pad image to square while keeping aspect ratio."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        pad_w = max_side - w
        pad_h = max_side - h
        padding = (pad_w // 2, pad_h // 2,
                   pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = ImageOps.expand(img, padding, fill=0)
        return img.resize((self.size, self.size), Image.Resampling.BILINEAR)


# ----------------------------
# Transforms
# ----------------------------
train_transform = transforms.Compose([
    PadToSquare(224),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    PadToSquare(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ----------------------------
# Dataset class
# ----------------------------
class DeepFashionDataset(Dataset):
    """
    DeepFashion Category and Attribute Prediction dataset.

    Uses official annotation files instead of folder structure.

    Args:
        img_root:        path to the img/ folder
        category_file:   path to Anno/list_category_img.txt
        partition_file:  path to Eval/list_eval_partition.txt
        split:           'train', 'val', or 'test'
        transform:       torchvision transforms
        min_images:      minimum images per class to include
        class_map:       optional external class map from training set
                         (pass this to val/test to ensure label consistency)
    """

    def __init__(self, img_root, category_file, partition_file,
                 split='train', transform=None, min_images=MIN_IMAGES,
                 class_map=None):

        self.img_root = img_root
        self.transform = transform
        self.split = split

        # Step 1: Load partition (which images belong to train/val/test)
        partition = {}
        with open(partition_file, 'r') as f:
            lines = f.readlines()[2:]  # skip header rows
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                partition[parts[0]] = parts[1]

        # Step 2: Load category labels
        img_labels = {}
        with open(category_file, 'r') as f:
            lines = f.readlines()[2:]  # skip header rows
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name = parts[0]
                label = int(parts[1]) - 1  # convert to 0-indexed
                img_labels[img_name] = label

        # Step 3: Count images per class in this split
        class_counts = {}
        for img_name, label in img_labels.items():
            if partition.get(img_name) == split:
                class_counts[label] = class_counts.get(label, 0) + 1

        # Step 4: Determine valid classes and build class map
        if class_map is not None:
            # Val/test: use the class map built from training set
            # This ensures labels are consistent across all splits
            self.class_map = class_map
            valid_classes = set(class_map.keys())
        else:
            # Train: build class map from scratch
            valid_classes = {
                label for label, count in class_counts.items()
                if count >= min_images
            }
            sorted_valid = sorted(valid_classes)
            self.class_map = {old: new for new, old in enumerate(sorted_valid)}

        # Step 5: Class names and count
        sorted_keys = sorted(self.class_map.keys())
        self.classes = [CATEGORY_NAMES[i] for i in sorted_keys]
        self.num_classes = len(self.class_map)

        # Step 6: Build final sample list
        self.samples = []
        for img_name, label in img_labels.items():
            if partition.get(img_name) == split and label in valid_classes:
                self.samples.append((img_name, self.class_map[label]))

        print(f"[{split}] {len(self.samples)} images | "
              f"{self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label