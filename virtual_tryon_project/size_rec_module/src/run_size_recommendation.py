#!/usr/bin/env python3
"""Main script to run the size recommendation pipeline."""

import argparse
import os
import json
from .mediapipe_utils import get_landmarks
from .midas_utils import get_depth_map
from .measurement_utils import estimate_measurements
from .size_matching import find_best_size

DEFAULT_SIZE_CHART_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_size_charts.json")

def main(args):
    """Runs the size recommendation pipeline."""

    print("Starting Size Recommendation Pipeline...")

    # --- 1. Input Acquisition (Paths provided via args) ---
    print(f"Using Front Image: {args.front_image}")
    print(f"Using Side Image: {args.side_image}")
    print(f"Metadata: Height={args.height}, Weight={args.weight}, Gender={args.gender}")

    # --- 2. Image Analysis --- 
    print("Analyzing front image...")
    front_landmarks_3d, front_landmarks_2d, front_segmentation = get_landmarks(args.front_image)
    front_depth_map = get_depth_map(args.front_image, args.midas_model_path, args.midas_model_type)
    # TODO: Add contour extraction using OpenCV if needed

    print("Analyzing side image...")
    side_landmarks_3d, side_landmarks_2d, side_segmentation = get_landmarks(args.side_image)
    side_depth_map = get_depth_map(args.side_image, args.midas_model_path, args.midas_model_type)
    # TODO: Add contour extraction using OpenCV if needed

    # --- 3. Measurement Estimation --- 
    print("Estimating body measurements...")
    estimated_measurements = estimate_measurements(
        front_landmarks_3d, front_depth_map, 
        side_landmarks_3d, side_depth_map,
        args.height, args.weight, args.gender
        # Add other inputs like 2D landmarks, contours, segmentation if the algorithm requires them
    )
    print(f"Estimated Measurements: {estimated_measurements}")

    # --- 4. Size Mapping --- 
    print(f"Loading size charts from: {args.size_chart_path}")
    try:
        with open(args.size_chart_path, 'r') as f:
            size_charts = json.load(f)
    except Exception as e:
        print(f"Error loading size charts: {e}")
        return

    print(f"Finding best size for garment type: {args.garment_type}")
    recommended_size = find_best_size(
        estimated_measurements, 
        size_charts, 
        args.gender, 
        args.garment_type
    )

    # --- 5. Output --- 
    print("--- Size Recommendation Result ---")
    if recommended_size:
        print(f"Recommended Size: {recommended_size}")
    else:
        print("Could not determine a suitable size.")
    print("----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clothing Size Recommendation Pipeline")
    parser.add_argument("--front_image", required=True, help="Path to the front view image.")
    parser.add_argument("--side_image", required=True, help="Path to the side view image.")
    parser.add_argument("--height", required=True, type=float, help="User height (e.g., in cm).")
    parser.add_argument("--weight", required=True, type=float, help="User weight (e.g., in kg).")
    parser.add_argument("--gender", required=True, choices=['male', 'female', 'unisex'], help="User gender.")
    parser.add_argument("--garment_type", required=True, help="Type of garment (e.g., 't-shirt', 'jeans'). Must match keys in size chart.")
    parser.add_argument("--size_chart_path", default=DEFAULT_SIZE_CHART_PATH, help="Path to the JSON file containing size charts.")
    parser.add_argument("--midas_model_path", default="../midas/weights", help="Path to the directory containing MiDaS model weights.")
    parser.add_argument("--midas_model_type", default="dpt_beit_large_512", help="Type of MiDaS model to use.")
    # Add arguments for MediaPipe model paths/options if needed
    
    args = parser.parse_args()
    main(args)

