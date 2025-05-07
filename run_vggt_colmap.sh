#!/bin/bash
set -e

# --- Argument Parsing ---
if [ "$#" -ne 4 ]; then # Changed to 4 arguments: input_type, input_path, output_base_dir, fps
    echo "Usage: $0 <input_type> <input_path_in_container> <output_base_dir_in_container> <fps_or_NA>"
    echo "  <input_type>: 'video' or 'frames_dir'"
    echo "  <input_path_in_container>: Path to the input video file or directory of frames."
    echo "  <output_base_dir_in_container>: Base directory where all outputs will be stored."
    echo "  <fps_or_NA>: Frames per second (if video) or 'NA' (if frames_dir)."
    exit 1
fi

INPUT_TYPE="$1"
INPUT_PATH_CONTAINER="$2" # Path to video OR directory of frames inside container
OUTPUT_BASE_DIR="$3"      # This is CONTAINER_OUTPUT_BASE_DIR from the docker run command
FPS_VALUE="$4"            # FPS for video, or "NA" if input is frames_dir

# Activate Python virtual environment
VENV_PATH="/opt/venv_vggt"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "INFO: Python virtual environment activated from ${VENV_PATH}"
else
    echo "ERROR: Python virtual environment not found at ${VENV_PATH}"
    exit 1
fi

# VGGT code location (where vggt_to_colmap.py is)
VGGT_REPO_DIR="/app/vggt"
# Our custom inference script location
CUSTOM_INFERENCE_SCRIPT="/app/run_vggt_inference.py"

# --- Directory Setup ---
# MOVIE_FILENAME will be derived differently based on input type
if [ "$INPUT_TYPE" == "video" ]; then
    MOVIE_BASENAME=$(basename "$INPUT_PATH_CONTAINER") # e.g., input.mp4
    MOVIE_FILENAME="${MOVIE_BASENAME%.*}"           # e.g., input
elif [ "$INPUT_TYPE" == "frames_dir" ]; then
    # For a frames directory, use its name as the base for other output folders
    MOVIE_FILENAME=$(basename "$INPUT_PATH_CONTAINER") # e.g., my_scene_frames
    # Let's try to make it cleaner, e.g. if it ends with _frames, remove it
    if [[ "$MOVIE_FILENAME" == *"_frames" ]]; then
        MOVIE_FILENAME="${MOVIE_FILENAME%_frames}"
    fi
else
    echo "ERROR: Invalid input_type: '${INPUT_TYPE}'. Must be 'video' or 'frames_dir'."
    exit 1
fi

# Frames will go into a subdirectory of OUTPUT_BASE_DIR if extracted from video
# If frames are provided, FRAMES_DIR will point to INPUT_PATH_CONTAINER
FRAMES_DIR_SUBPATH="${OUTPUT_BASE_DIR}/${MOVIE_FILENAME}_frames" # Default if we extract
COLMAP_DATASET_DIR="${OUTPUT_BASE_DIR}/${MOVIE_FILENAME}_colmap_data"

if [ "$INPUT_TYPE" == "video" ]; then
    FRAMES_DIR="$FRAMES_DIR_SUBPATH"
    mkdir -p "$FRAMES_DIR"
    FPS_TO_EXTRACT="$FPS_VALUE"
    if ! [[ "$FPS_TO_EXTRACT" =~ ^[0-9]+([\.][0-9]+)?$ && $(echo "$FPS_TO_EXTRACT > 0" | bc -l) -eq 1 ]]; then
        echo "ERROR: Invalid FPS value '$FPS_TO_EXTRACT' for video input."
        exit 1
    fi
else # frames_dir
    FRAMES_DIR="$INPUT_PATH_CONTAINER" # Use the provided directory of frames
    FPS_TO_EXTRACT="N/A (frames provided)"
fi

mkdir -p "$COLMAP_DATASET_DIR" # Create this regardless

echo "--- Pipeline Started (New Workflow) ---"
echo "Input Type:       $INPUT_TYPE"
echo "Input Path:       $INPUT_PATH_CONTAINER"
echo "Output Base Dir:  $OUTPUT_BASE_DIR (for results.pth.tar)"
echo "Frames Dir (used for VGGT): $FRAMES_DIR"
echo "COLMAP Dataset Dir: $COLMAP_DATASET_DIR"
echo "Extraction FPS:   $FPS_TO_EXTRACT"
echo "Python executable: $(which python)"
echo "PyTorch version:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HuggingFace Cache: ${HF_HOME}"
echo "------------------------"

# --- 1. Extract Frames using FFmpeg (only if input is video) ---
if [ "$INPUT_TYPE" == "video" ]; then
    echo "Step 1/3: Extracting frames from '$INPUT_PATH_CONTAINER' at ${FPS_TO_EXTRACT} FPS..."
    ffmpeg -i "$INPUT_PATH_CONTAINER" -vf "fps=${FPS_TO_EXTRACT}" -qscale:v 2 "${FRAMES_DIR}/frame_%06d.jpg" -loglevel error
    FRAME_COUNT=$(ls -1q "${FRAMES_DIR}" | wc -l)
    if [ "$FRAME_COUNT" -eq 0 ]; then
        echo "ERROR: FFmpeg failed to extract any frames."
        exit 1
    fi
    echo "INFO: Extracted $FRAME_COUNT frames to $FRAMES_DIR."
else
    echo "Step 1/3: Skipping frame extraction (frames provided in $FRAMES_DIR)."
    FRAME_COUNT=$(ls -1q "${FRAMES_DIR}" | wc -l) # Count existing frames
     if [ "$FRAME_COUNT" -eq 0 ]; then
        echo "ERROR: Provided frames directory '$FRAMES_DIR' is empty or contains no image files."
        exit 1
    fi
    echo "INFO: Using $FRAME_COUNT frames from $FRAMES_DIR."
fi

# --- 2. Run VGGT Inference using our custom script ---
# Output path for run_vggt_inference.py should be OUTPUT_BASE_DIR
# so results.pth.tar is in the parent directory of FRAMES_DIR (if frames were extracted)
# or in parent of the provided frames_dir (if frames were given)
# The vggt_to_colmap.py expects results.pth.tar in parent of its --image_dir
# If FRAMES_DIR is /workspace/output_data/my_scene_frames, its parent is /workspace/output_data
# So, --output_path for run_vggt_inference.py should be the directory that *will become* the parent of FRAMES_DIR.
# This is effectively OUTPUT_BASE_DIR.
echo "Step 2/3: Running VGGT inference using ${CUSTOM_INFERENCE_SCRIPT}..."
python "${CUSTOM_INFERENCE_SCRIPT}" \
    --input_path "$FRAMES_DIR" \
    --output_path "$OUTPUT_BASE_DIR" \
    # --model_hf_id "facebook/VGGT-1B" # Default in script
echo "INFO: VGGT inference complete. results.pth.tar should be in $OUTPUT_BASE_DIR."

if [ ! -f "${OUTPUT_BASE_DIR}/results.pth.tar" ]; then
    echo "ERROR: Inference script did not produce ${OUTPUT_BASE_DIR}/results.pth.tar"
    exit 1
fi

# --- 3. Convert VGGT Output to COLMAP Format ---
echo "Step 3/3: Converting VGGT output to COLMAP format..."
# vggt_to_colmap.py will look for results.pth.tar in parent of --image_dir
# It takes --output_dir for where to put the colmap sparse model etc.
python "${VGGT_REPO_DIR}/vggt_to_colmap.py" \
    --image_dir "$FRAMES_DIR" \
    --output_dir "$COLMAP_DATASET_DIR" \
    # Add other optional args for vggt_to_colmap.py if needed, e.g. --binary
echo "INFO: COLMAP conversion complete. Dataset ready in $COLMAP_DATASET_DIR."

echo "--- Pipeline Finished Successfully ---"
echo "Results in: ${OUTPUT_BASE_DIR}"
echo "  - Frames used by VGGT: ${FRAMES_DIR}"
echo "  - VGGT predictions: ${OUTPUT_BASE_DIR}/results.pth.tar"
echo "  - COLMAP dataset: ${COLMAP_DATASET_DIR}"
echo "--------------------------------------"