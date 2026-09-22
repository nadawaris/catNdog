"""
fetch_large_unseen_test.py
--------------------------
Downloads 100 completely new, unique Cat images and 100 completely new, unique Dog images
into test_unseen_large/cats and test_unseen_large/dogs.
Guarantees zero overlap with dataset/ through strict SHA-256 deduplication.
"""

import os
import sys
import io
import time
import hashlib
import requests
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATS_DIR = os.path.join(BASE_DIR, "dataset", "cats")
DOGS_DIR = os.path.join(BASE_DIR, "dataset", "dogs")
UNSEEN_DIR = os.path.join(BASE_DIR, "test_unseen_large")
UNSEEN_CATS = os.path.join(UNSEEN_DIR, "cats")
UNSEEN_DOGS = os.path.join(UNSEEN_DIR, "dogs")

os.makedirs(UNSEEN_CATS, exist_ok=True)
os.makedirs(UNSEEN_DOGS, exist_ok=True)


def get_all_existing_hashes():
    hashes = set()
    for directory in [CATS_DIR, DOGS_DIR, UNSEEN_CATS, UNSEEN_DOGS]:
        if os.path.exists(directory):
            for fname in os.listdir(directory):
                fpath = os.path.join(directory, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, "rb") as f:
                            hashes.add(hashlib.sha256(f.read()).hexdigest())
                    except Exception:
                        pass
    return hashes


def download_unseen_dogs(target_count=100):
    seen_hashes = get_all_existing_hashes()
    current = len([f for f in os.listdir(UNSEEN_DOGS) if f.endswith('.jpg')])
    print(f"[UNSEEN DOGS] Current count: {current}. Target: {target_count}")
    headers = {"User-Agent": "Mozilla/5.0"}

    while current < target_count:
        needed = min(50, target_count - current)
        try:
            r = requests.get(f"https://dog.ceo/api/breeds/image/random/{needed}", headers=headers, timeout=10)
            if r.status_code == 200:
                urls = r.json().get("message", [])
                for url in urls:
                    if current >= target_count:
                        break
                    try:
                        resp = requests.get(url, headers=headers, timeout=8)
                        if resp.status_code == 200 and len(resp.content) > 1024:
                            fhash = hashlib.sha256(resp.content).hexdigest()
                            if fhash in seen_hashes:
                                continue
                            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                            filepath = os.path.join(UNSEEN_DOGS, f"unseen_dog_{current+1:04d}.jpg")
                            img.save(filepath, "JPEG", quality=90)
                            seen_hashes.add(fhash)
                            current += 1
                    except Exception:
                        pass
            time.sleep(0.4)
        except Exception as e:
            print(f"Dog API error: {e}")
            break

    print(f"[UNSEEN DOGS COMPLETE] {current} unseen dog images in {UNSEEN_DOGS}")
    return current


def download_unseen_cats(target_count=100):
    seen_hashes = get_all_existing_hashes()
    current = len([f for f in os.listdir(UNSEEN_CATS) if f.endswith('.jpg')])
    print(f"[UNSEEN CATS] Current count: {current}. Target: {target_count}")
    headers = {"User-Agent": "Mozilla/5.0"}

    retries = 0
    while current < target_count and retries < 25:
        try:
            r = requests.get("https://api.thecatapi.com/v1/images/search?limit=10", headers=headers, timeout=8)
            if r.status_code == 200:
                items = r.json()
                for item in items:
                    url = item.get("url")
                    if not url or current >= target_count:
                        break
                    try:
                        resp = requests.get(url, headers=headers, timeout=8)
                        if resp.status_code == 200 and len(resp.content) > 1024:
                            fhash = hashlib.sha256(resp.content).hexdigest()
                            if fhash in seen_hashes:
                                continue
                            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                            filepath = os.path.join(UNSEEN_CATS, f"unseen_cat_{current+1:04d}.jpg")
                            img.save(filepath, "JPEG", quality=90)
                            seen_hashes.add(fhash)
                            current += 1
                    except Exception:
                        pass
            time.sleep(0.4)
        except Exception as e:
            retries += 1
            time.sleep(1)

    print(f"[UNSEEN CATS COMPLETE] {current} unseen cat images in {UNSEEN_CATS}")
    return current


if __name__ == "__main__":
    download_unseen_cats(100)
    download_unseen_dogs(100)
