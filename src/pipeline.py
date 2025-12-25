from vision.image_loader import load_images
from vision.tag_generator import generate_tags
from nlp.description_generator import generate_description
from trends.trend_predictor import predict_trends

def run_pipeline(image_folder):
    images = load_images(image_folder)

    for image in images:
        tags = generate_tags(image)
        description = generate_description(tags)
        trends = predict_trends()

        print("Image:", image)
        print("Tags:", tags)
        print("Description:", description)
        print("Trends:", trends)
        print("-" * 30)
