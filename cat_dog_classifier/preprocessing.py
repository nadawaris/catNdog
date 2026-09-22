"""
cat_dog_classifier / preprocessing.py
--------------------------------------
Single, unified, and reusable preprocessing module for:
- Training image dataset loading
- Validation & isolated test evaluation
- Single-image inference (CLI & Web API)
- External sanity testing

Guarantees 100% consistency across training and inference.
"""

import os
import io
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, Dict, Any

try:
    from config import IMAGE_SIZE, COLOR_MODE, INPUT_SHAPE
except ImportError:
    from cat_dog_classifier.config import IMAGE_SIZE, COLOR_MODE, INPUT_SHAPE


def load_and_preprocess_single_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """
    Standard preprocessing pipeline:
    1. Loads image from file path, raw bytes, or PIL Image object.
    2. Converts color space strictly to RGB (3 channels: Red, Green, Blue).
    3. Resizes using high-quality bilinear interpolation to target_size (default 128x128).
    4. Converts to float32 NumPy array.
    5. Normalizes pixel intensities strictly to [0.0, 1.0] by dividing by 255.0.
    
    Returns:
    --------
    np.ndarray of shape (128, 128, 3), dtype=np.float32, range in [0.0, 1.0]
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"[ERROR] Image file does not exist: {image_input}")
        pil_img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, io.BytesIO):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure RGB color mode (handles RGBA, Grayscale, CMYK, Palette formats)
    if pil_img.mode != COLOR_MODE:
        pil_img = pil_img.convert(COLOR_MODE)

    # Resize to standardized dimensions
    resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float32 array normalized to [0.0, 1.0]
    img_array = np.array(resized_img, dtype=np.float32) / 255.0

    return img_array


def preprocess_for_inference(
    image_input: Union[str, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> torch.Tensor:
    """
    Preprocesses a single image and adds the batch dimension for PyTorch model feed-forward.
    
    Returns:
    --------
    torch.Tensor of shape (1, 3, 128, 128), dtype=torch.float32
    """
    img_array = load_and_preprocess_single_image(image_input, target_size=target_size)
    ch_first = np.transpose(img_array, (2, 0, 1))
    tensor = torch.from_numpy(ch_first).unsqueeze(0).float()
    return tensor


def get_tensor_statistics(tensor: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
    """
    Inspects and returns debug statistics for a preprocessed tensor.
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min_val": float(np.min(arr)),
        "max_val": float(np.max(arr)),
        "mean_val": float(np.mean(arr)),
        "std_val": float(np.std(arr)),
        "is_normalized_0_1": bool(0.0 <= np.min(arr) and np.max(arr) <= 1.0),
        "has_nans": bool(np.isnan(arr).any())
    }


def print_tensor_statistics(tensor: np.ndarray, name: str = "Input Tensor"):
    """
    Prints tensor diagnostic metrics to stdout.
    """
    stats = get_tensor_statistics(tensor)
    print(f"\n[DEBUG TENSOR] {name}:")
    print(f"  Shape      : {stats['shape']}")
    print(f"  Dtype      : {stats['dtype']}")
    print(f"  Range      : [{stats['min_val']:.4f}, {stats['max_val']:.4f}] (Normalized: {stats['is_normalized_0_1']})")
    print(f"  Mean / Std : {stats['mean_val']:.4f} / {stats['std_val']:.4f}")
    print(f"  NaNs Found : {stats['has_nans']}\n")
