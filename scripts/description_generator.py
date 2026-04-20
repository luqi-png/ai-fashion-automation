import random

# ----------------------------
# Attribute type groupings
# ----------------------------
ATTR_TYPE_NAMES = {
    1: "texture",
    2: "fabric",
    3: "shape",
    4: "part",
    5: "style"
}

# Category to clothing type mapping for natural language
CATEGORY_TYPE = {
    "Anorak": "outerwear", "Blazer": "top", "Blouse": "top",
    "Bomber": "jacket", "Button-Down": "shirt", "Cardigan": "top",
    "Flannel": "shirt", "Halter": "top", "Henley": "top",
    "Hoodie": "sweatshirt", "Jacket": "jacket", "Jersey": "top",
    "Parka": "coat", "Peacoat": "coat", "Poncho": "top",
    "Sweater": "sweater", "Tank": "top", "Tee": "t-shirt",
    "Top": "top", "Turtleneck": "top", "Capris": "pants",
    "Chinos": "pants", "Culottes": "pants", "Cutoffs": "shorts",
    "Gauchos": "pants", "Jeans": "jeans", "Jeggings": "pants",
    "Jodhpurs": "pants", "Joggers": "pants", "Leggings": "leggings",
    "Sarong": "skirt", "Shorts": "shorts", "Skirt": "skirt",
    "Sweatpants": "pants", "Sweatshorts": "shorts", "Trunks": "shorts",
    "Caftan": "dress", "Cape": "outerwear", "Coat": "coat",
    "Coverup": "dress", "Dress": "dress", "Jumpsuit": "jumpsuit",
    "Kaftan": "dress", "Kimono": "robe", "Nightdress": "dress",
    "Onesie": "outfit", "Robe": "robe", "Romper": "romper",
    "Shirtdress": "dress", "Sundress": "dress"
}

# Style-related words that suggest occasion
OCCASION_KEYWORDS = {
    "casual":    ["casual", "relaxed", "everyday", "laid-back"],
    "formal":    ["formal", "tailored", "structured", "business"],
    "sporty":    ["sporty", "athletic", "active", "gym"],
    "bohemian":  ["boho", "bohemian", "flowy", "festival"],
    "elegant":   ["elegant", "sophisticated", "chic", "luxe"],
    "streetwear":["streetwear", "urban", "edgy", "oversized"]
}

# Fabric/texture words for description enrichment
FABRIC_WORDS = [
    "chiffon", "denim", "cotton", "silk", "lace", "velvet",
    "leather", "knit", "woven", "jersey", "linen", "satin",
    "mesh", "tweed", "corduroy", "suede", "fleece"
]

TEXTURE_WORDS = [
    "floral", "striped", "plaid", "checkered", "geometric",
    "abstract", "printed", "embroidered", "patterned", "solid",
    "textured", "ribbed", "quilted", "ruffled", "pleated"
]


def detect_occasion(tags):
    """Detect the likely occasion based on tags."""
    tag_set = set(t.lower() for t in tags)
    for occasion, keywords in OCCASION_KEYWORDS.items():
        if any(kw in tag_set for kw in keywords):
            return occasion
    return None


def split_tags_by_type(tags, attr_types_map):
    """Split tags into fabric, texture, shape, style groups."""
    fabrics = []
    textures = []
    shapes = []
    styles = []
    others = []

    for tag in tags:
        tag_lower = tag.lower()
        if any(f in tag_lower for f in FABRIC_WORDS):
            fabrics.append(tag)
        elif any(t in tag_lower for t in TEXTURE_WORDS):
            textures.append(tag)
        elif tag_lower in ["sleeveless", "long sleeve", "short sleeve",
                           "cropped", "oversized", "fitted", "loose"]:
            shapes.append(tag)
        elif tag_lower in ["casual", "formal", "sporty", "boho",
                           "elegant", "streetwear", "athletic"]:
            styles.append(tag)
        else:
            others.append(tag)

    return fabrics, textures, shapes, styles, others


def generate_description(category, tags, confidence):
    """
    Generate a natural language description from category and tags.

    Args:
        category:   predicted clothing category (e.g. "Blouse")
        tags:       list of attribute tag strings
        confidence: model confidence score (0-100)

    Returns:
        str: natural language description
    """
    clothing_type = CATEGORY_TYPE.get(category, category.lower())
    occasion = detect_occasion(tags)
    fabrics, textures, shapes, styles, others = split_tags_by_type(tags, {})

    # --- Build description parts ---
    parts = []

    # Opening — vary based on confidence
    if confidence >= 85:
        openers = [
            f"This is a {category.lower()}",
            f"A stylish {category.lower()}",
            f"Featuring a {category.lower()} design",
        ]
    elif confidence >= 60:
        openers = [
            f"This appears to be a {category.lower()}",
            f"Likely a {category.lower()}",
        ]
    else:
        openers = [
            f"This piece resembles a {category.lower()}",
            f"A garment that may be a {category.lower()}",
        ]

    opening = random.choice(openers)
    parts.append(opening)

    # Fabric mention
    if fabrics:
        parts.append(f"crafted in {fabrics[0]}")

    # Texture/pattern mention
    if textures:
        if len(textures) == 1:
            parts.append(f"with a {textures[0]} pattern")
        else:
            parts.append(f"featuring {textures[0]} and {textures[1]} detailing")

    # Shape/silhouette mention
    if shapes:
        parts.append(f"in a {shapes[0]} silhouette")

    # Other notable attributes
    if others:
        notable = others[:2]
        if len(notable) == 1:
            parts.append(f"with {notable[0]} details")
        else:
            parts.append(f"with {notable[0]} and {notable[1]} details")

    # Occasion/style closing
    if occasion:
        occasion_phrases = {
            "casual":     "perfect for everyday wear",
            "formal":     "ideal for formal or professional settings",
            "sporty":     "great for active or athletic occasions",
            "bohemian":   "suited for a free-spirited, bohemian look",
            "elegant":    "ideal for elegant or evening occasions",
            "streetwear": "suited for an urban streetwear aesthetic"
        }
        parts.append(occasion_phrases[occasion])
    elif styles:
        parts.append(f"with a {styles[0]} aesthetic")
    else:
        # Generic closing based on category type
        closings = {
            "dress":      "making it a versatile wardrobe piece",
            "top":        "easy to style for any occasion",
            "pants":      "offering a comfortable and stylish fit",
            "jacket":     "adding a polished finish to any outfit",
            "outerwear":  "providing warmth with style",
            "jeans":      "a wardrobe staple for casual looks",
            "shorts":     "ideal for warm weather styling",
            "skirt":      "a feminine and versatile piece",
        }
        closing = closings.get(clothing_type, "a great addition to any wardrobe")
        parts.append(closing)

    # Join parts into a sentence
    if len(parts) == 1:
        description = parts[0] + "."
    else:
        description = parts[0] + ", " + ", ".join(parts[1:-1])
        if len(parts) > 2:
            description += ", " + parts[-1] + "."
        else:
            description += " " + parts[-1] + "."

    # Capitalize first letter
    description = description[0].upper() + description[1:]

    return description


# ----------------------------
# Standalone test
# ----------------------------
if __name__ == '__main__':
    test_cases = [
        {
            "category": "Blouse",
            "tags": ["pleated", "sheer", "chiffon", "floral", "sleeveless"],
            "confidence": 99.72
        },
        {
            "category": "Dress",
            "tags": ["floral", "casual", "cotton", "midi"],
            "confidence": 88.5
        },
        {
            "category": "Jacket",
            "tags": ["leather", "studded", "edgy", "streetwear"],
            "confidence": 76.3
        },
        {
            "category": "Jeans",
            "tags": ["denim", "ripped", "casual", "slim"],
            "confidence": 91.0
        },
        {
            "category": "Sweater",
            "tags": ["knit", "ribbed", "oversized", "casual"],
            "confidence": 65.4
        }
    ]

    print("--- Description Generator Test ---\n")
    for case in test_cases:
        desc = generate_description(
            case["category"],
            case["tags"],
            case["confidence"]
        )
        print(f"Category:    {case['category']} ({case['confidence']}%)")
        print(f"Tags:        {', '.join(case['tags'])}")
        print(f"Description: {desc}")
        print()
