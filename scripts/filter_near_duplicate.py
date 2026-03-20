import cv2
import os
import shutil
from pathlib import Path
from PIL import Image
import imagehash

def batch_filter_incidents(root_dir, output_root, threshold=5):
    """
    Iterates through incident folders and filters near-duplicates using perceptual hashing.
    Threshold: Hamming distance (0-64). Lower = more similar. Default 5 means very similar images.
    """
    root_path = Path(root_dir)
    output_path = Path(output_root)

    # Get all subdirectories (incidents)
    incidents = [d for d in root_path.iterdir() if d.is_dir()]
    
    total_before = 0
    total_after = 0
    
    for incident in incidents:
        print(f"\n--- Processing Incident: {incident.name} ---")
        save_dir = output_path / incident.name
        save_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([f for f in incident.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not images:
            continue
        
        total_before += len(images)

        # Initialize first image
        last_path = str(images[0])
        last_img = Image.open(last_path)
        last_hash = imagehash.phash(last_img)
        
        shutil.copy(last_path, save_dir / images[0].name)
        count = 1

        for i in range(1, len(images)):
            curr_path = str(images[i])
            curr_img = Image.open(curr_path)
            curr_hash = imagehash.phash(curr_img)

            # Calculate Hamming distance between hashes
            distance = curr_hash - last_hash

            if distance > threshold:
                shutil.copy(curr_path, save_dir / images[i].name)
                last_hash = curr_hash
                count += 1
            
        total_after += count
        print(f"Finished {incident.name}: Kept {count}/{len(images)} images.")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Total images before: {total_before}, after: {total_after}")
    print(f"Removed {total_before - total_after} near-duplicates ({100*(1-total_after/total_before):.1f}% reduction)")
    print(f"{'='*60}\n")

# Update these paths
input_data = "/home/robby/workspace/Oil-Spill-Benz/datasets/raw/dv5_new"
output_data = "/home/robby/workspace/Oil-Spill-Benz/datasets/processed/dv5_unique"

batch_filter_incidents(input_data, output_data)