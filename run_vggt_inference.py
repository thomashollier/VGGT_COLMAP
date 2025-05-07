import argparse
import torch
import os
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
import time # For basic timing

# Attempt to import VGGT and its utilities
try:
    from vggt.models.vggt import VGGT
except ImportError:
    print("ERROR: VGGT package not found. Make sure it's installed and in PYTHONPATH.")
    exit(1)

def load_and_preprocess_images_custom(image_paths, device):
    print(f"INFO: Starting image loading and preprocessing for {len(image_paths)} images...")
    start_time = time.time()
    target_size = 504 
    preprocess = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    batch_images = []
    for i, img_path in enumerate(image_paths):
        try:
            # Simple progress for image loading
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(image_paths):
                 print(f"INFO: Loading and preprocessing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            image = Image.open(img_path).convert("RGB")
            processed_image = preprocess(image)
            batch_images.append(processed_image)
        except Exception as e:
            print(f"WARNING: Could not load or process image {img_path}: {e}")
            continue
            
    if not batch_images:
        print("ERROR: No images were successfully loaded and processed.")
        return None
    
    end_time = time.time()
    print(f"INFO: Image loading and preprocessing finished in {end_time - start_time:.2f} seconds.")
    return torch.stack(batch_images).to(device)

def main():
    parser = argparse.ArgumentParser(description="Run VGGT inference on a directory of images.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the directory to save inference results (results.pth.tar).")
    parser.add_argument("--model_hf_id", type=str, default="facebook/VGGT-1B", help="Hugging Face model ID for VGGT.")
    args = parser.parse_args()

    print("--- VGGT Inference Script Started ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Using device: {device}")

    dtype = torch.float32
    if device == "cuda":
        try:
            if torch.cuda.get_device_capability()[0] >= 8: 
                dtype = torch.bfloat16
                print("INFO: Using bfloat16 for CUDA.")
            else: 
                dtype = torch.float16
                print("INFO: Using float16 for CUDA.")
        except Exception: 
            dtype = torch.float16
            print("INFO: Defaulting to float16 on CUDA (capability check failed).")
    else: 
        print("INFO: Using float32 on CPU.")

    print(f"INFO: Initializing model '{args.model_hf_id}'. This may involve downloading from Hugging Face Hub if not cached...")
    # Hugging Face Hub will print download progress if it's downloading.
    # We can check if the model is cached, but it's a bit involved to do reliably for all cases.
    # The from_pretrained method itself handles caching.
    model_load_start_time = time.time()
    try:
        model = VGGT.from_pretrained(args.model_hf_id).to(device)
        model.eval() 
    except Exception as e:
        print(f"ERROR: Failed to initialize or load model from Hugging Face: {e}")
        print("Ensure an internet connection if downloading for the first time, and the model ID is correct.")
        print("Hugging Face Hub cache location is typically ~/.cache/huggingface/hub or what HF_HOME points to.")
        exit(1)
    model_load_end_time = time.time()
    print(f"INFO: Model initialized in {model_load_end_time - model_load_start_time:.2f} seconds.")

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_path, ext)))
    image_files.sort() 

    if not image_files:
        print(f"ERROR: No images found in '{args.input_path}' with extensions {image_extensions}")
        exit(1)
    print(f"INFO: Found {len(image_files)} images to process in '{args.input_path}'.")

    images_tensor = load_and_preprocess_images_custom(image_files, device)
    if images_tensor is None: exit(1)

    print(f"INFO: Running VGGT model inference on a batch of {images_tensor.shape[0]} images...")
    inference_start_time = time.time()
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_tensor)
        else: 
            predictions = model(images_tensor) 
    inference_end_time = time.time()
    print(f"INFO: Model inference finished in {inference_end_time - inference_start_time:.2f} seconds.")

    print(f"DEBUG: Prediction keys from model: {list(predictions.keys())}")
    
    results_to_save = {} 
    
    # --- Key Adaptation ---
    print("INFO: Adapting model predictions for saving...")
    if 'pose_enc' in predictions:
        results_to_save['pred_cameras'] = predictions['pose_enc'] 
        print(f"  INFO: Mapping 'pose_enc' to 'pred_cameras'. Type: {type(predictions['pose_enc'])}")
    else:
        print("WARNING: 'pose_enc' (for camera poses) not found in predictions.")

    if 'world_points' in predictions: 
        results_to_save['pred_points_maps'] = predictions['world_points'].cpu()
        print(f"  INFO: Mapping 'world_points' to 'pred_points_maps'. Original shape: {predictions['world_points'].shape}")
    else:
        print("WARNING: 'world_points' (for point maps) not found in predictions.")

    if 'depth' in predictions: 
        results_to_save['pred_depths'] = predictions['depth'].cpu()
        print(f"  INFO: Mapping 'depth' to 'pred_depths'. Original shape: {predictions['depth'].shape}")
    else:
        print("WARNING: 'depth' (for depth maps) not found in predictions.")

    if 'world_points_conf' in predictions:
        results_to_save['pred_points_confs'] = predictions['world_points_conf'].cpu()
        print(f"  INFO: Saving 'world_points_conf' as 'pred_points_confs'. Original shape: {predictions['world_points_conf'].shape}")
    if 'depth_conf' in predictions:
        results_to_save['pred_depths_confs'] = predictions['depth_conf'].cpu()
        print(f"  INFO: Saving 'depth_conf' as 'pred_depths_confs'. Original shape: {predictions['depth_conf'].shape}")
    # --- End Key Adaptation ---

    target_size_for_meta = 504 
    if images_tensor is not None:
        B, C, H, W = images_tensor.shape 
        results_to_save['meta_info'] = { 
            'image_names': [os.path.basename(p) for p in image_files],
            'input_hw': [H, W] 
        }
    else: 
        # This case should ideally not be reached if images are loaded
        results_to_save['meta_info'] = {
            'image_names': [os.path.basename(p) for p in image_files],
            'input_hw': [target_size_for_meta, target_size_for_meta] 
        }
    print(f"INFO: Added 'meta_info' with {len(results_to_save['meta_info']['image_names'])} images and input_hw: {results_to_save['meta_info']['input_hw']}")

    # Ensure output directory exists (it should, created by the shell script)
    if not os.path.exists(args.output_path):
        print(f"WARNING: Output path directory '{args.output_path}' does not exist. This is unexpected.")
        # Attempt to create it, though the calling script should handle this.
        # os.makedirs(args.output_path, exist_ok=True) 

    output_file_path = os.path.join(args.output_path, "results.pth.tar")
    
    print(f"INFO: Preparing to save results to '{output_file_path}'...")
    try:
        # Ensure all tensors in results_to_save are on CPU before saving
        if 'pred_cameras' in results_to_save:
            cam_data = results_to_save['pred_cameras']
            if hasattr(cam_data, 'cpu'):
                results_to_save['pred_cameras'] = cam_data.cpu()
                print("  INFO: Moved 'pred_cameras' (tensor) to CPU for saving.")
            elif isinstance(cam_data, list):
                new_pred_cameras = []
                moved_list_item_to_cpu = False
                for i, cam_item in enumerate(cam_data):
                    if hasattr(cam_item, 'cpu'):
                        new_pred_cameras.append(cam_item.cpu())
                        moved_list_item_to_cpu = True
                    else:
                        new_pred_cameras.append(cam_item) 
                results_to_save['pred_cameras'] = new_pred_cameras
                if moved_list_item_to_cpu:
                    print("  INFO: Moved 'pred_cameras' (list of tensors/items) to CPU for saving.")
            else:
                print(f"  INFO: 'pred_cameras' is of type {type(cam_data)}, not moving to CPU unless it has .cpu().")


        torch.save(results_to_save, output_file_path)
        print(f"INFO: Inference results successfully saved to '{output_file_path}'.")
    except Exception as e:
        print(f"ERROR: Failed to save results: {e}")
        print("DEBUG: Contents of results_to_save that failed to save:")
        for key, value in results_to_save.items():
            value_info = f"Type: {type(value)}"
            if hasattr(value, 'shape'):
                value_info += f", Shape: {value.shape}"
            if hasattr(value, 'device'):
                value_info += f", Device: {value.device}"
            print(f"  Key: '{key}', {value_info}")
        exit(1)
    
    print("--- VGGT Inference Script Finished Successfully ---")

if __name__ == "__main__":
    main()