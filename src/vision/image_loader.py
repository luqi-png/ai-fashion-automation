import os

def load_images(folder_path):
    """
    Loads image file paths from a directory.
    """
    images = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            images.append(os.path.join(folder_path, file))
    return images
