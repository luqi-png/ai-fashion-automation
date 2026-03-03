from torchvision import transforms
from PIL import Image, ImageOps

class PadToSquare:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        pad_w = max_side - w
        pad_h = max_side - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = ImageOps.expand(img, padding, fill=0)
        return img.resize((self.size, self.size), Image.BILINEAR)

train_transform = transforms.Compose([
    PadToSquare(300),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    PadToSquare(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])