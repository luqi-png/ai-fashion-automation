import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms
from PIL import ImageOps

# ----------------------------
# Paths — update these
# ----------------------------
IMG_ROOT       = r"C:\Users\ranal\ai-fashion-automation\data\img_highres\img"
CATEGORY_FILE  = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_category_img.txt"
PARTITION_FILE = r"C:\Users\ranal\ai-fashion-automation\data\Eval\list_eval_partition.txt"
ATTR_CLOTH_FILE= r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_attr_cloth.txt"
ATTR_IMG_FILE  = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_attr_img.txt"
MODEL_PATH     = r"C:\Users\ranal\ai-fashion-automation\models\efficientnet_b0_best.pth"

# Minimum images to include a class (must match training)
MIN_IMAGES = 100

# How many top attributes to return per image
TOP_K_ATTRS = 10

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

# Attribute type names for context
ATTR_TYPE_NAMES = {
    1: "texture",
    2: "fabric",
    3: "shape",
    4: "part",
    5: "style"
}

# ----------------------------
# PadToSquare transform
# ----------------------------
class PadToSquare:
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

transform = transforms.Compose([
    PadToSquare(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Step 1: Load attribute names
# ----------------------------
def load_attribute_names(attr_cloth_file):
    attr_names = []
    attr_types = []
    with open(attr_cloth_file, 'r') as f:
        lines = f.readlines()[2:]  # skip count + header
    for line in lines:
        parts = line.strip().rsplit(None, 1)
        if len(parts) == 2:
            attr_names.append(parts[0].strip())
            attr_types.append(int(parts[1].strip()))
    return attr_names, attr_types

# ----------------------------
# Step 2: Load per-image attributes
# ----------------------------
def load_image_attributes(attr_img_file):
    print("Loading attribute annotations (this may take a moment)...")
    img_attrs = {}
    with open(attr_img_file, 'r') as f:
        lines = f.readlines()[2:]  # skip count + header
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 1:
            img_name = parts[0]
            # 1 = positive attribute, -1 = negative, 0 = unknown
            attrs = [int(x) for x in parts[1:]]
            img_attrs[img_name] = attrs
    print(f"Loaded attributes for {len(img_attrs)} images")
    return img_attrs

# ----------------------------
# Step 3: Load class map from training set
# ----------------------------
def load_train_class_map(category_file, partition_file, min_images):
    # Load partition
    partition = {}
    with open(partition_file, 'r') as f:
        lines = f.readlines()[2:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            partition[parts[0]] = parts[1]

    # Load category labels
    img_labels = {}
    with open(category_file, 'r') as f:
        lines = f.readlines()[2:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            img_labels[parts[0]] = int(parts[1]) - 1

    # Count per class in train split
    class_counts = {}
    for img_name, label in img_labels.items():
        if partition.get(img_name) == 'train':
            class_counts[label] = class_counts.get(label, 0) + 1

    # Build class map
    valid_classes = {l for l, c in class_counts.items() if c >= min_images}
    sorted_valid = sorted(valid_classes)
    class_map = {old: new for new, old in enumerate(sorted_valid)}
    classes = [CATEGORY_NAMES[i] for i in sorted_valid]
    return class_map, classes

# ----------------------------
# Step 4: Load EfficientNet-B0 model
# ----------------------------
def load_model(model_path, num_classes, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path, weights_only=True,
                                      map_location=device))
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Step 5: Tag a single image
# ----------------------------
def tag_image(img_path, model, class_map, classes, img_attrs,
              attr_names, attr_types, device, top_k=TOP_K_ATTRS):

    # Normalize path to match annotation format
    # Annotations use forward slashes: img/ClassName/img_00000001.jpg
    norm_path = img_path.replace('\\', '/')

    # Find the img/ part in the path
    idx = norm_path.find('img/')
    if idx != -1:
        lookup_key = norm_path[idx:]
    else:
        lookup_key = norm_path

    # --- Classify image ---
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    predicted_category = classes[pred_idx]

    # --- Get attributes ---
    tags = []
    if lookup_key in img_attrs:
        attrs = img_attrs[lookup_key]
        for i, val in enumerate(attrs):
            if val == 1 and i < len(attr_names):  # positive attribute
                tags.append({
                    'name': attr_names[i],
                    'type': ATTR_TYPE_NAMES.get(attr_types[i], 'unknown')
                })
        # Limit to top_k
        tags = tags[:top_k]
    else:
        tags = []

    return {
        'category': predicted_category,
        'confidence': round(confidence * 100, 2),
        'tags': tags,
        'tag_names': [t['name'] for t in tags]
    }

# ----------------------------
# Main — demo on sample images
# ----------------------------
if __name__ == '__main__':
    from deepfashion_dataset import DeepFashionDataset

    Image.MAX_IMAGE_PIXELS = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load everything
    attr_names, attr_types = load_attribute_names(ATTR_CLOTH_FILE)
    print(f"Loaded {len(attr_names)} attribute names")

    img_attrs = load_image_attributes(ATTR_IMG_FILE)

    class_map, classes = load_train_class_map(
        CATEGORY_FILE, PARTITION_FILE, MIN_IMAGES
    )
    print(f"Classes: {classes}")

    model = load_model(MODEL_PATH, len(classes), device)
    print("Model loaded")

    # Test on 5 sample images from the dataset
    print("\n--- Sample Predictions ---\n")
    count = 0
    import sys
    sys.path.append(os.path.dirname(__file__))

    with open(PARTITION_FILE, 'r') as f:
        partition_lines = f.readlines()[2:]

    
    for line in partition_lines:
        parts = line.strip().split()
        if len(parts) == 2 and parts[1] == 'test':
            img_name = parts[0]
            img_path = os.path.join(IMG_ROOT, img_name.replace('img/', '', 1))

            if not os.path.exists(img_path):
                continue

            result = tag_image(
                img_path, model, class_map, classes,
                img_attrs, attr_names, attr_types, device
            )

            print(f"Image:      {img_name}")
            print(f"Category:   {result['category']} ({result['confidence']}% confidence)")
            print(f"Tags:       {', '.join(result['tag_names']) if result['tag_names'] else 'No tags found'}")
            print()

            count += 1
            if count >= 5:
                break

    print("Done.")