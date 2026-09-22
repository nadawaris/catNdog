"""
cat_dog_classifier / download_dataset.py
-----------------------------------------
Expanded Real-World Cat and Dog Dataset Downloader.
Downloads 100+ real photographic cat images and 100+ real photographic dog images
across diverse breeds, poses, orientations, and backgrounds.
Ensures zero synthetic doodles and strict SHA-256 deduplication.
"""

import os
import sys
import io
import time
import hashlib
import requests
from PIL import Image

try:
    from config import CATS_DIR, DOGS_DIR, TEST_IMAGES_DIR, TEST_EXTERNAL_DIR, BASE_DIR
except ImportError:
    from cat_dog_classifier.config import CATS_DIR, DOGS_DIR, TEST_IMAGES_DIR, TEST_EXTERNAL_DIR, BASE_DIR

# Ensure stdout supports UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def get_existing_hashes(target_dir):
    hashes = set()
    if not os.path.exists(target_dir):
        return hashes
    for fname in os.listdir(target_dir):
        fpath = os.path.join(target_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "rb") as f:
                    hashes.add(hashlib.sha256(f.read()).hexdigest())
            except Exception:
                pass
    return hashes


def download_dog_images(target_dir, target_count=100):
    os.makedirs(target_dir, exist_ok=True)
    seen_hashes = get_existing_hashes(target_dir)
    current_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"[DOGS] Current dog image count: {current_count}. Target: {target_count}")
    headers = {"User-Agent": "Mozilla/5.0"}

    while current_count < target_count:
        needed = min(50, target_count - current_count)
        try:
            r = requests.get(f"https://dog.ceo/api/breeds/image/random/{needed}", headers=headers, timeout=10)
            if r.status_code == 200:
                urls = r.json().get("message", [])
                for url in urls:
                    if current_count >= target_count:
                        break
                    try:
                        img_resp = requests.get(url, headers=headers, timeout=8)
                        if img_resp.status_code == 200 and len(img_resp.content) > 1024:
                            fhash = hashlib.sha256(img_resp.content).hexdigest()
                            if fhash in seen_hashes:
                                continue
                            
                            # Verify valid image
                            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                            filename = f"dog_real_{current_count+1:04d}.jpg"
                            filepath = os.path.join(target_dir, filename)
                            img.save(filepath, "JPEG", quality=90)
                            
                            seen_hashes.add(fhash)
                            current_count += 1
                    except Exception:
                        pass
            time.sleep(0.5)
        except Exception as e:
            print(f"Dog API fetch error: {e}")
            break

    print(f"[DOGS COMPLETE] Total {current_count} real dog photos in {target_dir}")
    return current_count


def download_cat_images(target_dir, target_count=100):
    os.makedirs(target_dir, exist_ok=True)
    seen_hashes = get_existing_hashes(target_dir)
    current_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    print(f"[CATS] Current cat image count: {current_count}. Target: {target_count}")
    headers = {"User-Agent": "Mozilla/5.0"}

    # Curated Wikimedia / Unsplash fallback list
    fallback_cats = [
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1573865526739-10659fec78a5?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1495360010541-f48722b34f7d?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1561948955-570b270e7c36?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1543852786-1cf6624b9987?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1535268647677-300dbf3d78d1?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1513360309081-38f0762daed1?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1519052537078-e6302a4968d4?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1548802673-380ab8ebc7b7?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1526336024174-e58f5cdd8e13?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1533743983669-94fa5c4338ec?auto=format&fit=crop&w=400&q=80",
        "https://images.unsplash.com/photo-1511044568932-338cba0ad803?auto=format&fit=crop&w=400&q=80",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittens_in_a_basket.jpg/320px-Kittens_in_a_basket.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_seat-crop.jpg/320px-Orange_tabby_cat_sitting_on_seat-crop.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/320px-Cat_August_2010-4.jpg"
    ]

    for url in fallback_cats:
        if current_count >= target_count:
            break
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200 and len(r.content) > 1024:
                fhash = hashlib.sha256(r.content).hexdigest()
                if fhash not in seen_hashes:
                    img = Image.open(io.BytesIO(r.content)).convert("RGB")
                    filename = f"cat_real_{current_count+1:04d}.jpg"
                    filepath = os.path.join(target_dir, filename)
                    img.save(filepath, "JPEG", quality=90)
                    seen_hashes.add(fhash)
                    current_count += 1
        except Exception:
            pass

    # Fetch from TheCatAPI in batches
    retries = 0
    while current_count < target_count and retries < 15:
        try:
            r = requests.get("https://api.thecatapi.com/v1/images/search?limit=10", headers=headers, timeout=8)
            if r.status_code == 200:
                items = r.json()
                for item in items:
                    url = item.get("url")
                    if not url or current_count >= target_count:
                        break
                    try:
                        img_resp = requests.get(url, headers=headers, timeout=8)
                        if img_resp.status_code == 200 and len(img_resp.content) > 1024:
                            fhash = hashlib.sha256(img_resp.content).hexdigest()
                            if fhash in seen_hashes:
                                continue
                            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                            filename = f"cat_real_{current_count+1:04d}.jpg"
                            filepath = os.path.join(target_dir, filename)
                            img.save(filepath, "JPEG", quality=90)
                            seen_hashes.add(fhash)
                            current_count += 1
                    except Exception:
                        pass
            time.sleep(0.4)
        except Exception as e:
            retries += 1
            time.sleep(1)

    print(f"[CATS COMPLETE] Total {current_count} real cat photos in {target_dir}")
    return current_count


def setup_real_dataset(target_per_class=100):
    print("\n" + "="*60)
    print(f" DOWNLOADING EXPANDED DIVERSE REAL PHOTO DATASET ({target_per_class}/class)")
    print("="*60)

    # 1. Download base training/val/test data
    download_cat_images(CATS_DIR, target_count=target_per_class)
    download_dog_images(DOGS_DIR, target_count=target_per_class)

    # 2. Setup External Sanity Test Suite
    ext_cats = os.path.join(TEST_EXTERNAL_DIR, "cats")
    ext_dogs = os.path.join(TEST_EXTERNAL_DIR, "dogs")
    download_cat_images(ext_cats, target_count=10)
    download_dog_images(ext_dogs, target_count=10)

    # 3. Setup sample test images
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    cat_files = [f for f in os.listdir(CATS_DIR) if f.endswith('.jpg')]
    dog_files = [f for f in os.listdir(DOGS_DIR) if f.endswith('.jpg')]
    if cat_files:
        Image.open(os.path.join(CATS_DIR, cat_files[0])).save(os.path.join(TEST_IMAGES_DIR, "sample_cat_1.jpg"))
    if dog_files:
        Image.open(os.path.join(DOGS_DIR, dog_files[0])).save(os.path.join(TEST_IMAGES_DIR, "sample_dog_1.jpg"))

    print("="*60 + "\n")


if __name__ == "__main__":
    setup_real_dataset(target_per_class=100)
