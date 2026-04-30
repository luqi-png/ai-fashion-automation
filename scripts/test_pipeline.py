"""
Unit tests for AI Fashion Automation pipeline.
Run with: python -m pytest scripts/test_pipeline.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(__file__))

from description_generator import (
    generate_description,
    detect_occasion,
    split_tags_by_type,
    CATEGORY_TYPE
)


# ─────────────────────────────────────────────
# Test Suite 1: Description Generator
# ─────────────────────────────────────────────
class TestDescriptionGenerator(unittest.TestCase):

    def test_high_confidence_opening(self):
        """High confidence (>=85%) should use assertive language."""
        desc = generate_description("Blouse", ["chiffon"], 90.0)
        self.assertTrue(
            desc.startswith("This is a") or desc.startswith("A stylish") or
            desc.startswith("Featuring"),
            f"Expected assertive opening, got: {desc}"
        )

    def test_medium_confidence_opening(self):
        """Medium confidence (60-84%) should use hedged language."""
        desc = generate_description("Blouse", ["chiffon"], 70.0)
        self.assertTrue(
            "appears" in desc.lower() or "likely" in desc.lower(),
            f"Expected hedged language, got: {desc}"
        )

    def test_low_confidence_opening(self):
        """Low confidence (<60%) should use tentative language."""
        desc = generate_description("Blouse", ["chiffon"], 45.0)
        self.assertTrue(
            "resembles" in desc.lower() or "may be" in desc.lower(),
            f"Expected tentative language, got: {desc}"
        )

    def test_description_ends_with_period(self):
        """All descriptions should end with a period."""
        desc = generate_description("Dress", ["floral", "casual"], 88.0)
        self.assertTrue(desc.endswith("."), f"Description should end with '.': {desc}")

    def test_description_starts_uppercase(self):
        """All descriptions should start with an uppercase letter."""
        desc = generate_description("Jacket", ["leather"], 75.0)
        self.assertTrue(desc[0].isupper(), f"Description should start uppercase: {desc}")

    def test_fabric_included_in_description(self):
        """Fabric words should appear in the description."""
        desc = generate_description("Blouse", ["chiffon", "pleated"], 92.0)
        self.assertIn("chiffon", desc.lower(),
                      f"Expected fabric 'chiffon' in description: {desc}")

    def test_empty_tags(self):
        """Description should still generate with no tags."""
        desc = generate_description("Dress", [], 80.0)
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 0)

    def test_category_in_description(self):
        """Category name should appear in description."""
        desc = generate_description("Jeans", ["denim", "casual"], 85.0)
        self.assertIn("jeans", desc.lower(),
                      f"Expected 'jeans' in description: {desc}")

    def test_all_categories_generate(self):
        """All 35 categories should produce a valid description."""
        categories = list(CATEGORY_TYPE.keys())
        for cat in categories:
            desc = generate_description(cat, ["casual"], 80.0)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 10,
                               f"Description too short for category {cat}: {desc}")


# ─────────────────────────────────────────────
# Test Suite 2: Occasion Detection
# ─────────────────────────────────────────────
class TestOccasionDetection(unittest.TestCase):

    def test_casual_detected(self):
        """Casual tags should be detected correctly."""
        occasion = detect_occasion(["casual", "cotton"])
        self.assertEqual(occasion, "casual")

    def test_formal_detected(self):
        """Formal tags should be detected correctly."""
        occasion = detect_occasion(["formal", "tailored"])
        self.assertEqual(occasion, "formal")

    def test_sporty_detected(self):
        """Sporty tags should be detected correctly."""
        occasion = detect_occasion(["sporty", "athletic"])
        self.assertEqual(occasion, "sporty")

    def test_no_occasion(self):
        """Unknown tags should return None."""
        occasion = detect_occasion(["pleated", "chiffon"])
        self.assertIsNone(occasion)

    def test_empty_tags(self):
        """Empty tags list should return None."""
        occasion = detect_occasion([])
        self.assertIsNone(occasion)


# ─────────────────────────────────────────────
# Test Suite 3: Tag Classification
# ─────────────────────────────────────────────
class TestTagClassification(unittest.TestCase):

    def test_fabric_detected(self):
        """Fabric words should be classified correctly."""
        fabrics, _, _, _, _ = split_tags_by_type(
            ["chiffon", "denim", "cotton"], {}
        )
        self.assertIn("chiffon", fabrics)
        self.assertIn("denim", fabrics)

    def test_texture_detected(self):
        """Texture words should be classified correctly."""
        _, textures, _, _, _ = split_tags_by_type(
            ["floral", "striped", "pleated"], {}
        )
        self.assertIn("floral", textures)
        self.assertIn("striped", textures)

    def test_shape_detected(self):
        """Shape/silhouette words should be classified correctly."""
        _, _, shapes, _, _ = split_tags_by_type(
            ["sleeveless", "oversized"], {}
        )
        self.assertIn("sleeveless", shapes)

    def test_mixed_tags(self):
        """Mixed tag types should all be classified."""
        fabrics, textures, shapes, styles, others = split_tags_by_type(
            ["chiffon", "floral", "sleeveless", "casual", "zip-front"], {}
        )
        self.assertIn("chiffon", fabrics)
        self.assertIn("floral", textures)
        self.assertIn("sleeveless", shapes)


# ─────────────────────────────────────────────
# Test Suite 4: PadToSquare Transform
# ─────────────────────────────────────────────
class TestPadToSquare(unittest.TestCase):

    def setUp(self):
        from image_tagger import PadToSquare
        self.transform = PadToSquare(224)

    def test_landscape_becomes_square(self):
        """Landscape image should become square."""
        img = Image.new("RGB", (300, 200))
        result = self.transform(img)
        self.assertEqual(result.size, (224, 224))

    def test_portrait_becomes_square(self):
        """Portrait image should become square."""
        img = Image.new("RGB", (200, 300))
        result = self.transform(img)
        self.assertEqual(result.size, (224, 224))

    def test_square_stays_square(self):
        """Square image should remain square."""
        img = Image.new("RGB", (300, 300))
        result = self.transform(img)
        self.assertEqual(result.size, (224, 224))

    def test_output_size_correct(self):
        """Output size should always match target size."""
        for w, h in [(100, 200), (400, 150), (224, 224), (50, 50)]:
            img = Image.new("RGB", (w, h))
            result = self.transform(img)
            self.assertEqual(result.size, (224, 224),
                             f"Failed for input size ({w}, {h})")


# ─────────────────────────────────────────────
# Test Suite 5: Model Architecture
# ─────────────────────────────────────────────
class TestModelArchitecture(unittest.TestCase):

    def test_efficientnet_output_shape(self):
        """Model should output correct number of classes."""
        from torchvision import models
        num_classes = 35
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, num_classes),
                         f"Expected output shape (1, 35), got {output.shape}")

    def test_softmax_sums_to_one(self):
        """Softmax probabilities should sum to 1."""
        from torchvision import models
        num_classes = 35
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.softmax(output, dim=1)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_dropout_disabled_in_eval(self):
        """Dropout should be disabled in eval mode."""
        from torchvision import models
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 35)
        )
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = model(dummy_input)
        self.assertTrue(torch.allclose(out1, out2),
                        "Eval mode outputs should be deterministic")


# ─────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────
if __name__ == '__main__':
    unittest.main(verbosity=2)
