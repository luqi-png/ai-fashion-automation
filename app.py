import os
import sys
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image

# Add scripts folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from image_tagger import (
    load_attribute_names,
    load_image_attributes,
    load_train_class_map,
    load_model,
    tag_image,
    transform
)
from description_generator import generate_description

# ----------------------------
# Paths — update these
# ----------------------------
IMG_ROOT        = r"C:\\Users\\ranal\\ai-fashion-automation\data\\img_highres\\img"
CATEGORY_FILE   = r"C:\\Users\\ranal\\ai-fashion-automation\data\Anno_coarse\\list_category_img.txt"
PARTITION_FILE  = r"C:\\Users\\ranal\\ai-fashion-automation\data\\Eval\\list_eval_partition.txt"
ATTR_CLOTH_FILE = r"C:\\Users\\ranal\\ai-fashion-automation\data\Anno_coarse\\list_attr_cloth.txt"
ATTR_IMG_FILE   = r"C:\\Users\\ranal\\ai-fashion-automation\data\Anno_coarse\\list_attr_img.txt"
MODEL_PATH      = r"C:\\Users\\ranal\\ai-fashion-automation\\models\\efficientnet_b0_best.pth"
MIN_IMAGES      = 100

UPLOAD_FOLDER   = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ----------------------------
# Load pipeline at startup
# ----------------------------
import torch
print("Loading pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

attr_names, attr_types = load_attribute_names(ATTR_CLOTH_FILE)
img_attrs = load_image_attributes(ATTR_IMG_FILE)
class_map, classes = load_train_class_map(CATEGORY_FILE, PARTITION_FILE, MIN_IMAGES)
model = load_model(MODEL_PATH, len(classes), device)
print(f"Pipeline ready — {len(classes)} classes")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------------------
# Routes
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    uploaded_image = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'No file selected.'
        else:
            file = request.files['image']
            if file.filename == '':
                error = 'No file selected.'
            elif not allowed_file(file.filename):
                error = 'Invalid file type. Please upload a JPG, PNG or WEBP image.'
            else:
                try:
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    uploaded_image = f'uploads/{filename}'

                    # Run pipeline
                    tag_result = tag_image(
                        filepath, model, class_map, classes,
                        img_attrs, attr_names, attr_types, device
                    )
                    description = generate_description(
                        tag_result['category'],
                        tag_result['tag_names'],
                        tag_result['confidence']
                    )

                    result = {
                        'category': tag_result['category'],
                        'confidence': tag_result['confidence'],
                        'tags': tag_result['tag_names'],
                        'description': description
                    }

                except Exception as e:
                    error = f'An error occurred: {str(e)}'

    return render_template('index.html',
                           result=result,
                           error=error,
                           uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
