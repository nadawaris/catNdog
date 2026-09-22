"""
cat_dog_classifier / download_dataset.py
-----------------------------------------
Automated dataset helper script.
Populates `dataset/cats/` and `dataset/dogs/` with sample images if a local dataset is missing.
"""

import os
import sys
import urllib.request
import numpy as np
from PIL import Image, ImageDraw

# Ensure stdout supports UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def create_dataset_structure(base_dir="dataset"):
    """Creates dataset/cats and dataset/dogs folders if missing."""
    cats_dir = os.path.join(base_dir, "cats")
    dogs_dir = os.path.join(base_dir, "dogs")
    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)
    return cats_dir, dogs_dir


def download_sample_images(base_dir="dataset", num_per_class=50):
    """
    Downloads or generates sample cat and dog images for testing and demonstration.
    """
    cats_dir, dogs_dir = create_dataset_structure(base_dir)

    existing_cats = [f for f in os.listdir(cats_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    existing_dogs = [f for f in os.listdir(dogs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"[DATASET] Current status: {len(existing_cats)} Cat images, {len(existing_dogs)} Dog images.")

    if len(existing_cats) >= 20 and len(existing_dogs) >= 20:
        print("[OK] Sufficient dataset already available!")
        return

    print("[INFO] Fetching/Generating sample Cat and Dog image dataset...")

    # Public domain sample images (Wikimedia Commons)
    cat_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittens_in_a_basket.jpg/320px-Kittens_in_a_basket.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_seat-crop.jpg/320px-Orange_tabby_cat_sitting_on_seat-crop.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/320px-Cat_August_2010-4.jpg"
    ]

    dog_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collesticker.jpg/320px-Collesticker.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/German_Shepherd_Dog_header.jpg/320px-German_Shepherd_Dog_header.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Golden_retriever_flat-coated.jpg/320px-Golden_retriever_flat-coated.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Golden_Retriever_multiview.jpg/320px-Golden_Retriever_multiview.jpg"
    ]

    def fetch_urls(urls, target_dir, prefix):
        downloaded = 0
        for idx, url in enumerate(urls):
            filepath = os.path.join(target_dir, f"{prefix}_web_{idx+1}.jpg")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as resp, open(filepath, 'wb') as f:
                    f.write(resp.read())
                img = Image.open(filepath).convert('RGB')
                img.resize((128, 128)).save(filepath)
                downloaded += 1
            except Exception as e:
                pass
        return downloaded

    print("  Downloading public domain image samples...")
    cats_dl = fetch_urls(cat_urls, cats_dir, "cat")
    dogs_dl = fetch_urls(dog_urls, dogs_dir, "dog")

    # Generate synthetic diverse features to expand dataset
    def generate_synthetic_samples(target_dir, prefix, label_type, count_needed):
        current_count = len([f for f in os.listdir(target_dir) if f.startswith(prefix)])
        np.random.seed(42 if label_type == 'cat' else 99)

        for i in range(current_count, count_needed):
            img = Image.new('RGB', (128, 128), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            bg_r = np.random.randint(180, 255)
            bg_g = np.random.randint(180, 255)
            bg_b = np.random.randint(180, 255)
            draw.rectangle([0, 0, 128, 128], fill=(bg_r, bg_g, bg_b))

            if label_type == 'cat':
                # CAT FEATURES: Pointy triangular ears, slit eyes, whiskers, orange/tabby coat
                color = (230, 120, 40) if i % 2 == 0 else (120, 120, 120)
                draw.ellipse([34, 38, 94, 98], fill=color)
                draw.polygon([(34, 48), (24, 18), (54, 38)], fill=(200, 80, 30))
                draw.polygon([(94, 48), (104, 18), (74, 38)], fill=(200, 80, 30))
                draw.ellipse([48, 54, 58, 66], fill=(50, 200, 50))
                draw.ellipse([70, 54, 80, 66], fill=(50, 200, 50))
                draw.polygon([(60, 72), (68, 72), (64, 76)], fill=(255, 150, 150))
                draw.line([(64, 76), (40, 78)], fill=(0, 0, 0), width=1)
                draw.line([(64, 76), (88, 78)], fill=(0, 0, 0), width=1)
            else:
                # DOG FEATURES: Floppy round ears, round dark eyes, broad snout, golden/brown coat
                color = (180, 110, 50) if i % 2 == 0 else (50, 50, 50)
                draw.ellipse([30, 30, 98, 98], fill=color)
                draw.ellipse([18, 40, 40, 90], fill=(130, 70, 30))
                draw.ellipse([88, 40, 110, 90], fill=(130, 70, 30))
                draw.ellipse([46, 50, 58, 62], fill=(20, 20, 20))
                draw.ellipse([70, 50, 82, 62], fill=(20, 20, 20))
                draw.ellipse([48, 65, 80, 88], fill=(230, 210, 190))
                draw.ellipse([58, 68, 70, 76], fill=(10, 10, 10))

            noise = np.random.randint(-15, 15, (128, 128, 3))
            arr = np.array(img, dtype=np.int16) + noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            final_img = Image.fromarray(arr)

            filepath = os.path.join(target_dir, f"{prefix}_{i+1}.jpg")
            final_img.save(filepath, quality=95)

    generate_synthetic_samples(cats_dir, "cat", "cat", num_per_class)
    generate_synthetic_samples(dogs_dir, "dog", "dog", num_per_class)

    # Generate separate test images
    test_dir = os.path.join(base_dir, "..", "test_images") if os.path.basename(base_dir) == "dataset" else "test_images"
    os.makedirs(test_dir, exist_ok=True)
    generate_synthetic_samples(test_dir, "sample_cat", "cat", 2)
    generate_synthetic_samples(test_dir, "sample_dog", "dog", 2)

    total_cats = len([f for f in os.listdir(cats_dir) if f.endswith('.jpg')])
    total_dogs = len([f for f in os.listdir(dogs_dir) if f.endswith('.jpg')])
    print(f"[DATASET READY] {total_cats} Cats in dataset/cats/, {total_dogs} Dogs in dataset/dogs/")


if __name__ == "__main__":
    download_sample_images(num_per_class=50)
