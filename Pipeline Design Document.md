# Pipeline Design Document

## 1. Overview

This document outlines the design for an integrated system providing:
1.  **Virtual Clothing Try-On:** Based on the IDM-VTON model.
2.  **Clothing Size Recommendation:** Based on user images, MediaPipe, OpenCV, and MiDaS depth estimation.

## 2. Selected Core Components

*   **Virtual Try-On:** [yisol/IDM-VTON](https://github.com/yisol/IDM-VTON) (Official Implementation)
*   **Depth Estimation (for Size Rec):** [isl-org/MiDaS](https://github.com/isl-org/MiDaS)
*   **Pose Estimation & Body Landmarks (for Size Rec):** [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)
*   **Image Processing (General):** OpenCV

## 3. Component Analysis

### 3.1 IDM-VTON (Virtual Try-On)

*   **Goal:** Generate an image of a person wearing a specific garment.
*   **Inputs:**
    *   Person Image (Model Image)
    *   Garment Image (Clothing Image)
    *   (Optional) Garment Mask
    *   (Optional) DensePose map of the person
*   **Key Internal Steps/Modules (as per paper/repo):**
    *   **Preprocessing:** Human Parsing (SCHP), DensePose generation (Detectron2), OpenPose (optional, mentioned in user req but less prominent in repo README).
    *   **Feature Extraction:** Garment features (potentially UNet-like, IP-Adapter with CLIP mentioned).
    *   **Try-On Network:** Diffusion model (SDXL based, fine-tuned IDM-VTON) performs inpainting/generation in latent space, guided by garment features and pose information.
    *   **VAE:** Encoding/Decoding images to/from latent space.
*   **Outputs:** Synthesized try-on image.
*   **Dependencies:** PyTorch, Accelerate, Diffusers, Transformers, Detectron2, OpenCV, various other ML/utility libraries (see `environment.yaml`). Requires specific pre-trained models (IP-Adapter, Human Parsing, DensePose, OpenPose, SDXL base if not included).
*   **Datasets:** Trained/Tested on VITON-HD, DressCode.
*   **Interface:** Command-line scripts (`inference.py`, `inference_dc.py`), Gradio demo (`gradio_demo/app.py`).
*   **Resource Requirements:** High VRAM GPU (README mentions >=16GB for ComfyUI adaptation, likely similar or higher for official repo training/inference), significant storage for models and datasets.

### 3.2 MiDaS (Depth Estimation)

*   **Goal:** Estimate depth map from a single RGB image.
*   **Inputs:** Single RGB image.
*   **Key Internal Steps/Modules:** Deep learning model (various backbones available, e.g., DPT, BEiT, Swin).
*   **Outputs:** Relative or metric depth map.
*   **Dependencies:** PyTorch, OpenCV, timm (see `environment.yaml`). Requires downloaded pre-trained MiDaS model weights.
*   **Interface:** Command-line script (`run.py`), PyTorch Hub.
*   **Resource Requirements:** GPU recommended for faster inference, various model sizes available trading off accuracy/speed.

### 3.3 MediaPipe (Pose/Landmarks)

*   **Goal:** Detect human pose landmarks (skeleton, face, hands) and potentially segmentation masks.
*   **Inputs:** RGB image or video frame.
*   **Key Internal Steps/Modules:** Pre-trained ML models for pose, face mesh, holistic tracking.
*   **Outputs:** Landmark coordinates (2D/3D), segmentation masks.
*   **Dependencies:** Python package (`mediapipe`).
*   **Interface:** Python API.
*   **Resource Requirements:** Runs on CPU, faster with GPU.

### 3.4 OpenCV (Image Processing)

*   **Goal:** General image loading, manipulation, contour detection, geometric calculations.
*   **Inputs:** Images, video frames.
*   **Outputs:** Processed images, numerical data (contours, measurements).
*   **Dependencies:** Python package (`opencv-python`).
*   **Interface:** Python API.
*   **Resource Requirements:** Primarily CPU-bound.

## 4. Proposed Integrated Pipeline Architecture

*(To be detailed in the next step)*




## 4. Proposed Integrated Pipeline Architecture

This system will consist of two primary modules, potentially integrated into a single user interface but functionally distinct:

**Module 1: Virtual Try-On (Based on IDM-VTON)**

1.  **Input Acquisition:**
    *   User provides a full-body image (`person_image.jpg`).
    *   User provides a garment image (`garment_image.jpg`).
2.  **Preprocessing Pipeline:**
    *   **Human Parsing:** Run SCHP model (using provided checkpoints) on `person_image.jpg` to generate a segmentation mask (`person_mask.png`).
    *   **DensePose Estimation:** Run Detectron2 (using provided checkpoints) on `person_image.jpg` to generate a DensePose map (`person_densepose.png`).
    *   **OpenPose Estimation (Optional):** Run OpenPose (using provided checkpoints) if required by the specific IDM-VTON configuration being used.
    *   **Garment Masking:** Generate a mask for the garment (`garment_mask.png`), potentially using simple thresholding/background removal or a dedicated segmentation model if needed.
    *   **(IDM-VTON Internal):** The model likely generates an agnostic representation of the person (removing existing clothing based on parsing/pose).
3.  **IDM-VTON Core Inference:**
    *   Load pre-trained models: IDM-VTON diffusion model, VAE, IP-Adapter, CLIP image encoder.
    *   Prepare inputs for the diffusion model: Latent representation of agnostic person image, garment image features (via IP-Adapter/CLIP), pose information (DensePose), text prompts (if applicable, e.g., garment description).
    *   Run the diffusion sampling process (inpainting the garment onto the person representation in latent space).
    *   Decode the resulting latent representation using the VAE.
4.  **Output:**
    *   Synthesized image (`tryon_result.jpg`) showing the person wearing the garment.

**Module 2: Size Recommendation (Based on MediaPipe, MiDaS, OpenCV)**

1.  **Input Acquisition:**
    *   User provides a front-view image (`front_image.jpg`).
    *   User provides a side-view image (`side_image.jpg`).
    *   User provides metadata: `height` (cm/in), `weight` (kg/lb), `gender`. *(Assumption: These are provided. Alternatively, height could be estimated if an object of known size is in frame, but this adds complexity).* 
2.  **Image Analysis Pipeline (Run for both front and side images):**
    *   **Pose & Landmarks:** Run MediaPipe Pose and Holistic models to extract 3D body landmarks (`landmarks_3d.json`) and potentially 2D landmarks (`landmarks_2d.json`) and segmentation masks (`segmentation_mask.png`).
    *   **Depth Estimation:** Run MiDaS (e.g., `dpt_beit_large_512` for quality) to generate a depth map (`depth_map.pfm` or `.png`).
    *   **Contour Extraction:** Use OpenCV on the input image or segmentation mask to find body contours (`contours.json`).
3.  **Measurement Estimation:**
    *   **Calibration:** Use the provided `height` and potentially landmark positions (e.g., distance between specific landmarks in 3D space) to establish a scale (pixels/real-world units).
    *   **Measurement Calculation:** Combine 3D landmarks, depth map information, and contour data from both views. Apply geometric algorithms to estimate key body measurements (e.g., chest circumference, waist circumference, hip circumference, shoulder width, inseam length). This requires careful algorithm design, potentially referencing anthropological measurement techniques adapted for computer vision.
    *   Store estimated measurements (`measurements.json`).
4.  **Size Mapping:**
    *   Load brand-specific size charts (`size_charts.json` - *Note: This database needs to be created/sourced*).
    *   Based on `gender` and estimated `measurements`, compare against the size charts for relevant garment types (e.g., tops, bottoms).
    *   Implement a matching logic (e.g., find the size where most measurements fit within the range, potentially prioritizing key measurements like chest/waist).
5.  **Output:**
    *   Recommended size (e.g., `
