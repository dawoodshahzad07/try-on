# Virtual Try-On & Size Recommendation Project

## 1. Introduction

This project provides a framework for implementing:

1.  **Virtual Clothing Try-On:** Utilizing the state-of-the-art IDM-VTON model to visualize garments on user images.
2.  **Clothing Size Recommendation:** Estimating user body measurements from images using computer vision techniques (MediaPipe, MiDaS, OpenCV) and recommending appropriate clothing sizes based on brand charts.

This project integrates existing powerful open-source models and provides a structure for running both pipelines. 

**Disclaimer:** 
*   The underlying models (IDM-VTON, MiDaS) require significant computational resources, particularly high-VRAM GPUs (potentially 16GB+ for IDM-VTON).
*   The size recommendation module, specifically the `measurement_utils.py` and `mediapipe_utils.py` components, contains **placeholder logic**. Implementing accurate 3D body measurement estimation from images is a complex research problem and requires significant development effort beyond the scope of this initial project assembly.
*   The accuracy of both modules depends heavily on the quality of input images, the underlying models, and the completeness of the size charts.

## 2. Project Structure

```
/virtual_tryon_project
|-- docs/                  # Documentation files (like this one, potentially others)
|   |-- pipeline_design.md # Detailed design document
|-- tryon_module/          # Virtual Try-On component
|   |-- IDM-VTON/          # Cloned official IDM-VTON repository
|       |-- assets/
|       |-- ckpt/          # Pre-trained models for IDM-VTON go here
|       |-- configs/
|       |-- gradio_demo/
|       |-- ip_adapter/
|       |-- preprocess/
|       |-- src/
|       |-- environment.yaml # Conda environment for IDM-VTON
|       |-- inference.py
|       |-- inference_dc.py
|       |-- train_xl.py
|       |-- README.md      # Original IDM-VTON README
|       |-- ... (other IDM-VTON files)
|-- size_rec_module/       # Size Recommendation component
|   |-- midas/             # Cloned official MiDaS repository
|   |   |-- weights/       # Pre-trained MiDaS models go here
|   |   |-- run.py
|   |   |-- environment.yaml # Original MiDaS environment file
|   |   |-- README.md      # Original MiDaS README
|   |   |-- ... (other MiDaS files)
|   |-- src/
|   |   |-- __init__.py
|   |   |-- run_size_recommendation.py # Main script for size recommendation
|   |   |-- mediapipe_utils.py         # Utilities for MediaPipe (Placeholder)
|   |   |-- midas_utils.py             # Utilities for MiDaS integration
|   |   |-- measurement_utils.py       # Utilities for measurement estimation (Placeholder)
|   |   |-- size_matching.py           # Utilities for size chart matching (Placeholder)
|   |-- data/
|   |   |-- sample_size_charts.json    # Example size chart data
|   |-- requirements.txt   # Python dependencies for this module
|-- README.md              # This file (Overall project guide)
```

## 3. Setup and Installation

**Prerequisites:**
*   Linux environment (tested on Ubuntu)
*   Anaconda or Miniconda installed
*   Git
*   NVIDIA GPU with CUDA installed (highly recommended, essential for IDM-VTON performance)

**Steps:**

1.  **Clone this Project:** If you haven't already, clone the entire project repository.

2.  **Setup IDM-VTON Environment:**
    *   Navigate to the IDM-VTON directory: `cd virtual_tryon_project/tryon_module/IDM-VTON`
    *   Create the Conda environment using the provided file: `conda env create -f environment.yaml`
    *   Activate the environment: `conda activate idm`
    *   **Install Detectron2:** Follow the official Detectron2 installation instructions carefully, as it often requires specific PyTorch/CUDA versions: [https://detectron2.readthedocs.io/en/latest/tutorials/install.html](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

3.  **Setup Size Recommendation Environment:**
    *   Navigate to the size recommendation module: `cd virtual_tryon_project/size_rec_module`
    *   It's recommended to use the *same* Conda environment (`idm`) if possible to avoid conflicts, especially with PyTorch. Alternatively, create a new environment and install dependencies.
    *   Install required packages: `pip install -r requirements.txt`
    *   **Install PyTorch/Torchvision:** Ensure you have PyTorch and Torchvision installed compatible with your CUDA version and the versions required by MiDaS and potentially MediaPipe. You might need to install these manually via `pip` or `conda` following PyTorch official instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   **Install MiDaS Dependencies (if needed):** The MiDaS `environment.yaml` lists its specific dependencies. If you are using a separate environment or the `idm` environment lacks something, install them (`pip install timm` etc.).

4.  **Download Pre-trained Models:**
    *   **IDM-VTON Models:**
        *   Follow the instructions in the `tryon_module/IDM-VTON/README.md` under the "Pretrained Models" section.
        *   Download the IP-Adapter models and place them in `tryon_module/IDM-VTON/ckpt/ip_adapter/`.
        *   Download the image encoder and place it in `tryon_module/IDM-VTON/ckpt/image_encoder/`.
        *   Download checkpoints for human parsing, DensePose, and OpenPose and place them under `tryon_module/IDM-VTON/ckpt/` in their respective subdirectories (`humanparsing`, `densepose`, `openpose`) as specified in the IDM-VTON README.
    *   **MiDaS Models:**
        *   Navigate to `virtual_tryon_project/size_rec_module/midas/`.
        *   Create the weights directory if it doesn't exist: `mkdir weights`
        *   Download the desired MiDaS model weights (e.g., `dpt_beit_large_512.pt`) from the links provided in the `size_rec_module/midas/README.md`.
        *   Place the downloaded `.pt` file(s) into the `size_rec_module/midas/weights/` directory.

5.  **Prepare Datasets (Optional - for IDM-VTON Training/Standard Inference):**
    *   If you intend to run the standard IDM-VTON inference scripts (`inference.py`, `inference_dc.py`) or training (`train_xl.py`), you need to download and structure the VITON-HD or DressCode datasets as described in the `tryon_module/IDM-VTON/README.md` under "Datasets".




## 4. Running the Pipelines

**Activate the Conda environment before running any scripts:** `conda activate idm`

### 4.1 Running the Virtual Try-On Module (IDM-VTON)

*   **Using the Gradio Demo (Recommended for basic testing):**
    *   Navigate to the demo directory: `cd virtual_tryon_project/tryon_module/IDM-VTON/gradio_demo`
    *   Ensure all prerequisite models (Human Parsing, DensePose, OpenPose, IP-Adapter) are correctly placed in the `ckpt` directory as per the setup instructions.
    *   Run the Gradio app: `python app.py`
    *   Access the web interface (usually http://127.0.0.1:7860) to upload person and garment images and see the results.
*   **Using the Inference Scripts (for batch processing or integration):**
    *   Navigate to the IDM-VTON root: `cd virtual_tryon_project/tryon_module/IDM-VTON`
    *   Prepare your input data (person images, garment images) and place them in a directory structure expected by the scripts (refer to the original IDM-VTON README and potentially the `inference.py` or `inference_dc.py` scripts for details on `--data_dir` structure).
    *   Run the inference script. Example for VITON-HD dataset structure:
        ```bash
        accelerate launch inference.py \
            --width 768 --height 1024 --num_inference_steps 30 \
            --output_dir "result" \
            --unpaired \
            --data_dir "/path/to/your/vitonhd/test" \
            --seed 42 \
            --test_batch_size 2 \
            --guidance_scale 2.0
        ```
    *   Adjust parameters like `--data_dir`, `--output_dir`, `--test_batch_size` as needed. Refer to `inference.sh` and the IDM-VTON README for more options.

### 4.2 Running the Size Recommendation Module

*   **Prerequisites:**
    *   Ensure the environment (`idm` or a dedicated one) is active and has all packages from `size_rec_module/requirements.txt` installed (including MediaPipe, OpenCV, PyTorch).
    *   Ensure MiDaS model weights are downloaded in `size_rec_module/midas/weights/`.
    *   **Crucially:** The placeholder functions in `mediapipe_utils.py` and `measurement_utils.py` need to be replaced with actual, functional implementations for pose estimation, landmark extraction, and measurement calculation.
    *   Prepare your input images (front view, side view) and user metadata (height, weight, gender).
    *   Ensure the `sample_size_charts.json` (or your custom version) exists and contains entries for the `gender` and `garment_type` you intend to test.
*   **Running the Script:**
    *   Navigate to the size recommendation source directory: `cd virtual_tryon_project/size_rec_module/src`
    *   Execute the main script with appropriate arguments:
        ```bash
        python run_size_recommendation.py \
            --front_image /path/to/your/front_view.jpg \
            --side_image /path/to/your/side_view.jpg \
            --height 175 \
            --weight 70 \
            --gender male \
            --garment_type t-shirt \
            --size_chart_path ../data/sample_size_charts.json \
            --midas_model_path ../midas/weights \
            --midas_model_type dpt_beit_large_512 
        ```
    *   Adjust paths, metadata, garment type, and model types as needed.

## 5. Deployment Considerations

Deploying this full system, especially the IDM-VTON module, is challenging due to its high resource requirements.

*   **Hardware:** A powerful server with high-end NVIDIA GPUs (e.g., A100, H100 with significant VRAM - 24GB, 40GB, 80GB+) is likely necessary for reasonable inference times with IDM-VTON.
*   **API Service:** Wrap the inference logic (both try-on and size recommendation) in a web framework (like Flask or FastAPI) to create API endpoints. This allows web or mobile applications to send requests (images, metadata) and receive results (try-on image, recommended size).
*   **Model Serving:** For performance and scalability, consider dedicated model serving platforms (like NVIDIA Triton Inference Server, TorchServe, or cloud-based solutions like SageMaker, Vertex AI Endpoints). These can optimize model loading and parallel execution.
*   **Preprocessing:** The preprocessing steps (human parsing, DensePose) can also be computationally intensive. These might need to run on GPU instances as well, potentially as separate microservices or integrated into the main API.
*   **Size Recommendation Module:** While less demanding than IDM-VTON, MiDaS and MediaPipe still benefit from GPU acceleration. The measurement estimation logic (once implemented) might be CPU-bound depending on the algorithms used.
*   **Containerization (Docker):** Package the application, dependencies, and models into Docker containers for easier deployment and environment consistency. You would need separate Dockerfiles for the IDM-VTON environment and potentially the size recommendation environment.
*   **Cloud Deployment:** Cloud platforms (AWS, GCP, Azure) offer GPU instances suitable for hosting such models. Services like Kubernetes can help manage container orchestration.
*   **Cost:** Running high-end GPU instances continuously can be expensive. Consider strategies like serverless GPU inference (if available and suitable) or scaling down resources during idle periods.

## 6. Further Development & Limitations

*   **Size Recommendation Accuracy:** The core challenge lies in implementing robust `measurement_utils.py` and `mediapipe_utils.py`. This requires significant R&D.
*   **Size Charts:** A comprehensive database of size charts for various brands and garment types is needed for the size recommendation to be practical.
*   **IDM-VTON Robustness:** While powerful, virtual try-on models can still struggle with complex poses, occlusions, or unusual garment types.
*   **User Interface:** This project provides the backend pipelines. A separate frontend (web or mobile app) would be needed for user interaction.

