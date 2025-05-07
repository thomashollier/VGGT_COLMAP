You got it! Since I can't directly offer a "download" button, the best way to do this is to provide the full, raw text of the README.md in a single code block. You can then:

1.  Click the "Copy code" button that usually appears at the top right of the code block.
2.  Paste the entire content into a new plain text file.
3.  Save that file as `README.md` in your project directory.

Here is the complete `README.md` content formatted for easy copying:

```markdown
# VGGT to COLMAP Dockerized Pipeline

This project provides a Dockerized pipeline to process input videos or image sequences using Meta's VGGT (Visual Geometry Grounded Transformer) model and then convert the output into a COLMAP-compatible dataset.

The pipeline performs the following steps:
1.  (Optional) Extracts frames from an input video file using FFmpeg.
2.  Runs inference using a pre-trained VGGT model (specifically `facebook/VGGT-1B`) on the frames to predict camera parameters, depth maps, and point maps.
3.  Converts the VGGT predictions into a format suitable for use with [COLMAP](https://colmap.github.io/).

## Prerequisites

*   **Docker:** Ensure Docker is installed and running on your system.
*   **NVIDIA GPU & NVIDIA Container Toolkit:** For GPU acceleration (highly recommended for VGGT inference), you need an NVIDIA GPU with appropriate drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured for Docker. This is typically handled by Docker Desktop on Windows with WSL2 backend if GPU support is enabled.
*   **WSL2 (for Windows users):** It's recommended to run this within a WSL2 environment for better Docker performance and Linux compatibility.
*   **Git:** To clone this repository (if you're distributing it this way) or for the Dockerfile to clone the VGGT repository.
*   **Bash Shell:** The provided helper script `process_video_with_vggt_docker.sh` is a bash script.

## Included Files

*   `Dockerfile`: Defines the Docker image with all necessary dependencies (CUDA, Python, PyTorch, VGGT, etc.).
*   `run_vggt_inference.py`: A custom Python script that loads images, runs the VGGT model using the Hugging Face `transformers` library, and saves the predictions.
*   `run_vggt_colmap.sh`: The main pipeline script that runs *inside* the Docker container. It handles frame extraction (if needed), calls `run_vggt_inference.py`, and then calls the `vggt_to_colmap.py` script from the original VGGT repository.
*   `process_video_with_vggt_docker.sh`: A helper bash script for your host machine to easily build and run the Docker container with the correct arguments and volume mounts.
*   `README.md`: This file.

## Setup

### 1. (Optional) Clone This Repository
If you are sharing these files via a Git repository:
```bash
git clone <your_repository_url>
cd <repository_directory>
```
Otherwise, ensure `Dockerfile`, `run_vggt_inference.py`, `run_vggt_colmap.sh`, and `process_video_with_vggt_docker.sh` are in the same directory.

### 2. Build the Docker Image
Navigate to the directory containing the `Dockerfile` and other scripts. Run the following command to build the Docker image. This process might take a considerable amount of time, especially the first time, as it downloads base images, dependencies, and clones the VGGT repository.

```bash
docker build -t vggt-colmap-processor .
```
The image will be tagged as `vggt-colmap-processor`. You can change this tag if you wish, but remember to update it in the `process_video_with_vggt_docker.sh` script as well.

## Usage

The `process_video_with_vggt_docker.sh` script is provided to simplify running the pipeline.

### Make the Script Executable
```bash
chmod +x process_video_with_vggt_docker.sh
```

### Running the Pipeline

The script accepts two mandatory arguments and one optional argument:
1.  **Input Path:** Absolute path to either an input video file OR a directory containing pre-extracted image frames.
2.  **Parent Output Directory:** Absolute path to a directory where all processing results for this input will be stored in a new, uniquely named sub-directory.
3.  **FPS (Optional):** Frames Per Second to extract if the input is a video file. Defaults to 25 if not specified. This argument is ignored if the input is a directory of frames.

**Syntax:**
```bash
./process_video_with_vggt_docker.sh <full_path_to_input> <full_path_to_parent_output_dir> [fps]
```

**Example 1: Processing a Video File**
This will process `my_scene.mp4`, extract frames at 10 FPS, and store all results in a new sub-directory under `/mnt/c/my_projects/colmap_data/`.
```bash
./process_video_with_vggt_docker.sh /mnt/c/videos/my_scene.mp4 /mnt/c/my_projects/colmap_data 10
```
Output will be in a directory like `/mnt/c/my_projects/colmap_data/my_scene_vggt_colmap_output_10fps/`.

**Example 2: Processing a Directory of Pre-extracted Frames**
This will process images from `/mnt/c/my_image_sequences/sequence_01/` and store results under `/mnt/c/my_projects/colmap_data/`. The FPS argument is not needed here (or will be ignored if provided).
```bash
./process_video_with_vggt_docker.sh /mnt/c/my_image_sequences/sequence_01 /mnt/c/my_projects/colmap_data
```
Output will be in a directory like `/mnt/c/my_projects/colmap_data/sequence_01_vggt_colmap_output/`.

### Hugging Face Model Cache
The first time you run the pipeline, the VGGT model (`facebook/VGGT-1B`) will be downloaded from Hugging Face Hub. This can take some time and disk space (several gigabytes).

The `process_video_with_vggt_docker.sh` script is configured to create a cache directory named `models` in the same location as the script itself. This cache (`./models/`) will be mounted into the Docker container at `/opt/hf_cache`. This means that after the first download, subsequent runs will use the cached model, speeding up initialization.

If you wish to change the location of this host-side cache, modify the `RELATIVE_HOST_HF_CACHE_DIR` variable at the top of the `process_video_with_vggt_docker.sh` script.

### Output Structure
For an input named `my_input`, the script will create an output directory (e.g., `my_input_vggt_colmap_output_Xfps/`) inside your specified parent output directory. This output directory will contain:

*   `my_input_frames/`: (Only if input was a video) Directory containing the extracted JPEG frames. If you provided a frames directory, this path will point to your original frames.
*   `results.pth.tar`: A PyTorch archive file containing the raw predictions from the VGGT model. This file is placed directly in the `my_input_vggt_colmap_output_Xfps/` directory.
*   `my_input_colmap_data/`: The COLMAP-compatible dataset.
    *   `images/`: Contains copies of (or symlinks to) the processed image frames.
    *   `sparse/0/`: Contains `cameras.bin`, `images.bin`, and `points3D.bin` which can be imported into COLMAP.

## Troubleshooting

*   **`exec format error` when running the container:** This usually means the shell scripts (`run_vggt_colmap.sh` or `process_video_with_vggt_docker.sh`) have incorrect line endings (Windows CRLF instead of Unix LF). The Dockerfile includes a `dos2unix` step for `run_vggt_colmap.sh`. Ensure `process_video_with_vggt_docker.sh` on your host also has Unix line endings if you encounter issues running it. Most modern text editors allow you to change line endings (e.g., VS Code, Notepad++).
*   **Docker GPU issues:** Ensure your NVIDIA drivers are up to date and the NVIDIA Container Toolkit is correctly installed and configured. Test with a simple CUDA sample in Docker (e.g., `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`).
*   **File not found errors inside the container:** Double-check volume mount paths in the `process_video_with_vggt_docker.sh` script and ensure the source paths on your host are correct and accessible.
*   **Hugging Face download issues:** Ensure the machine running Docker has internet access. If you are behind a proxy, you might need to configure Docker and/or set environment variables like `HTTP_PROXY` and `HTTPS_PROXY` within the Dockerfile or container.

## Further Development
*   The `run_vggt_inference.py` script currently adapts the output keys from the VGGT model to what `vggt_to_colmap.py` likely expects based on older conventions. The mapping of `pose_enc` to `pred_cameras` might need further refinement depending on the exact structure of `pose_enc` and the precise requirements of `vggt_to_colmap.py` for camera parameters.
*   More sophisticated batching could be added to `run_vggt_inference.py` if processing a very large number of frames causes memory issues.
```
