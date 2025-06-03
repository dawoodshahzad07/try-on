#!/usr/bin/env python3
"""Utility functions for MiDaS depth estimation."""

import os
import sys
import cv2
import torch
import numpy as np

# Add the MiDaS repo path to sys.path to allow importing
MIDAS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "size_rec_module", "midas"))
if MIDAS_PATH not in sys.path:
    sys.path.append(MIDAS_PATH)

# Conditional import based on path existence
try:
    from midas.model_loader import load_model
    # from midas.transforms import Resize, NormalizeImage, PrepareForNet
    # Note: Need to check MiDaS repo for exact transform classes/functions
    # If transforms are not directly importable, might need to copy/adapt them
    # or use torch hub approach if simpler.
    print(f"[midas_utils] Successfully imported MiDaS components from {MIDAS_PATH}")
    MIDAS_AVAILABLE = True
except ImportError as e:
    print(f"[midas_utils] Warning: Could not import MiDaS components from {MIDAS_PATH}. Depth estimation will not work. Error: {e}")
    MIDAS_AVAILABLE = False

def get_depth_map(image_path, model_path, model_type="dpt_beit_large_512"):
    """Placeholder function to get depth map using MiDaS.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the directory containing MiDaS model weights.
        model_type (str): Type of MiDaS model to load.

    Returns:
        numpy.ndarray: Depth map, or None if MiDaS is unavailable or fails.
    """
    print(f"[midas_utils] Processing image: {image_path} with model {model_type}")
    if not MIDAS_AVAILABLE:
        print("[midas_utils] MiDaS components not available. Cannot estimate depth.")
        return None

    try:
        # --- Device Selection ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[midas_utils] Using device: {device}")

        # --- Load Model ---
        # Adjust model_path to point to the specific weight file if needed by load_model
        # Example: model_weights = os.path.join(model_path, f"{model_type}.pt") 
        # Check the load_model function signature in the MiDaS repo.
        model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=False, height=None, square=False)
        print(f"[midas_utils] Loaded MiDaS model: {model_type}")

        # --- Load Image and Transform ---
        img = cv2.imread(image_path)
        if img is None:
            print(f"[midas_utils] Error: Could not read image {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (check MiDaS repo for correct transform pipeline)
        # Example placeholder transform usage:
        # input_batch = transform({"image": img})["image"]
        # This needs verification based on the actual MiDaS transform implementation.
        # Using a simplified transform for now:
        img_input = transform(img).to(device)
        print(f"[midas_utils] Image transformed, shape: {img_input.shape}")

        # --- Inference ---
        with torch.no_grad():
            prediction = model(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        print(f"[midas_utils] Depth map generated, shape: {depth_map.shape}")
        return depth_map

    except Exception as e:
        print(f"[midas_utils] Error during depth estimation: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage (for testing)
# if __name__ == "__main__":
#     # Create dummy image and paths for testing
#     dummy_image_path = "/path/to/your/test_image.jpg"
#     dummy_model_path = "../midas/weights" # Adjust path as needed
#     if os.path.exists(dummy_image_path):
#         depth = get_depth_map(dummy_image_path, dummy_model_path)
#         if depth is not None:
#             # Visualize or save the depth map
#             output_path = "depth_map_output.png"
#             # Normalize depth map for visualization
#             depth_min = depth.min()
#             depth_max = depth.max()
#             if depth_max - depth_min > 0:
#                 out = 255 * (depth - depth_min) / (depth_max - depth_min)
#             else:
#                 out = np.zeros(depth.shape, dtype=depth.dtype)
#             cv2.imwrite(output_path, out.astype("uint8"))
#             print(f"Depth map saved to {output_path}")
#     else:
#         print(f"Test image not found: {dummy_image_path}")

