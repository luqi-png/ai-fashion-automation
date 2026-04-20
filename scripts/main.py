import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

sys.path.append(os.path.dirname(__file__))
from image_tagger import (
    load_attribute_names,
    load_image_attributes,
    load_train_class_map,
    load_model,
    tag_image,
    PadToSquare,
    transform
)
from description_generator import generate_description

# ----------------------------
# Paths — update these
# ----------------------------
IMG_ROOT        = r"C:\Users\ranal\ai-fashion-automation\data\img_highres\img"
CATEGORY_FILE   = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_category_img.txt"
PARTITION_FILE  = r"C:\Users\ranal\ai-fashion-automation\data\Eval\list_eval_partition.txt"
ATTR_CLOTH_FILE = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_attr_cloth.txt"
ATTR_IMG_FILE   = r"C:\Users\ranal\ai-fashion-automation\data\Anno_coarse\list_attr_img.txt"
MODEL_PATH      = r"C:\Users\ranal\ai-fashion-automation\models\efficientnet_b0_best.pth"

MIN_IMAGES = 100
Image.MAX_IMAGE_PIXELS = None


# ----------------------------
# Pipeline class
# ----------------------------
class FashionPipeline:
    """
    End-to-end AI Fashion Automation pipeline.

    Given an image path, returns:
      - Predicted clothing category
      - Confidence score
      - Attribute tags
      - Natural language description
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        print("Loading pipeline components...")

        # Component 2: Load attribute data
        self.attr_names, self.attr_types = load_attribute_names(ATTR_CLOTH_FILE)
        self.img_attrs = load_image_attributes(ATTR_IMG_FILE)

        # Component 1: Load class map and model
        self.class_map, self.classes = load_train_class_map(
            CATEGORY_FILE, PARTITION_FILE, MIN_IMAGES
        )
        self.model = load_model(
            MODEL_PATH, len(self.classes), self.device
        )

        print(f"Pipeline ready — {len(self.classes)} classes, "
              f"{len(self.attr_names)} attributes")
        print()

    def run(self, img_path):
        """
        Run the full pipeline on a single image.

        Args:
            img_path: full path to the image file

        Returns:
            dict with keys: category, confidence, tags, description
        """
        if not os.path.exists(img_path):
            return {"error": f"Image not found: {img_path}"}

        # Component 1 + 2: Classify and tag
        result = tag_image(
            img_path,
            self.model,
            self.class_map,
            self.classes,
            self.img_attrs,
            self.attr_names,
            self.attr_types,
            self.device
        )

        # Component 3: Generate description
        description = generate_description(
            result["category"],
            result["tag_names"],
            result["confidence"]
        )

        result["description"] = description
        return result

    def print_result(self, result):
        """Pretty print the pipeline result."""
        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print("=" * 55)
        print(f"  Category   : {result['category']}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  Tags       : {', '.join(result['tag_names']) if result['tag_names'] else 'None found'}")
        print(f"  Description: {result['description']}")
        print("=" * 55)
        print()


# ----------------------------
# Main — run on sample images
# ----------------------------
if __name__ == '__main__':
    # pipeline = FashionPipeline()

    # # Test on 5 sample images from the test set
    # print("--- Running Full Pipeline on Sample Images ---\n")

    # with open(PARTITION_FILE, 'r') as f:
    #     partition_lines = f.readlines()[2:]

    # count = 0
    # for line in partition_lines:
    #     parts = line.strip().split()
    #     if len(parts) == 2 and parts[1] == 'test':
    #         img_name = parts[0]
    #         img_path = os.path.join(
    #             IMG_ROOT,
    #             img_name.replace('img/', '', 1)
    #         )

    #         if not os.path.exists(img_path):
    #             continue

    #         print(f"Image: {img_name}")
    #         result = pipeline.run(img_path)
    #         pipeline.print_result(result)

    #         count += 1
    #         if count >= 5:
    #             break

#----------------------------------------
# Main - Custome Image Path as Arguments
# ----------------------------------------

    if len(sys.argv) > 1:
        # Run on image passed as command line argument
        custom_path = sys.argv[1]
        pipeline = FashionPipeline()
        print(f"Testing on: {custom_path}\n")
        result = pipeline.run(custom_path)
        pipeline.print_result(result)

print("Pipeline complete.")
