import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_yolo_dataset(labels_source_dir, images_source_dir, output_base_dir):
    """
    Organize YOLO dataset by:
    1. Copying all .txt files from labels_source_dir to output_base_dir/labels
    2. Finding and copying corresponding original images to output_base_dir/images
    """
    
    # Create output directories
    output_labels_dir = os.path.join(output_base_dir, 'labels')
    output_images_dir = os.path.join(output_base_dir, 'images')
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    
    print(f"=== Organizing YOLO Dataset ===")
    print(f"Labels source: {labels_source_dir}")
    print(f"Images source: {images_source_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Output labels: {output_labels_dir}")
    print(f"Output images: {output_images_dir}")
    print()
    
    # Get all .txt files from labels directory
    txt_files = [f for f in os.listdir(labels_source_dir) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} .txt label files")
    
    # Get all subdirectories in images source
    subdirs = [d for d in os.listdir(images_source_dir) 
               if os.path.isdir(os.path.join(images_source_dir, d))]
    print(f"Found {len(subdirs)} subdirectories in image source")
    print()
    
    # Create a mapping of base_name -> image_path
    print("Building image file mapping...")
    image_map = {}
    for subdir in tqdm(subdirs, desc="Scanning subdirectories"):
        subdir_path = os.path.join(images_source_dir, subdir)
        files = os.listdir(subdir_path)
        
        for file in files:
            # Only look for image files (not .json)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                base_name = os.path.splitext(file)[0]
                image_map[base_name] = os.path.join(subdir_path, file)
    
    print(f"Found {len(image_map)} image files in total")
    print()
    
    # Process each .txt file
    print("Copying files...")
    success_count = 0
    missing_images = []
    
    for txt_file in tqdm(txt_files, desc="Processing"):
        base_name = os.path.splitext(txt_file)[0]
        
        # Copy .txt file to output labels directory
        src_txt = os.path.join(labels_source_dir, txt_file)
        dst_txt = os.path.join(output_labels_dir, txt_file)
        shutil.copy2(src_txt, dst_txt)
        
        # Find and copy corresponding image
        if base_name in image_map:
            src_image = image_map[base_name]
            # Get the extension from the source image
            _, ext = os.path.splitext(src_image)
            dst_image = os.path.join(output_images_dir, base_name + ext)
            shutil.copy2(src_image, dst_image)
            success_count += 1
        else:
            missing_images.append(base_name)
    
    print()
    print("=== Summary ===")
    print(f"Total .txt files processed: {len(txt_files)}")
    print(f"Successfully copied image-label pairs: {success_count}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print(f"\nWarning: Could not find images for {len(missing_images)} labels:")
        for name in missing_images[:10]:  # Show first 10
            print(f"  - {name}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    print()
    print(f"✅ Dataset organized successfully!")
    print(f"   Labels: {output_labels_dir}")
    print(f"   Images: {output_images_dir}")


if __name__ == "__main__":
    # Configuration
    labels_source_directory = r'/home/robby/workspace/Oil-Spill-Benz/datasets/raw/dv3-all/labels'
    images_source_directory = r'/home/robby/workspace/Oil-Spill-Benz/datasets/raw/DV3/Categorize_UAV'
    output_directory = r'/home/robby/workspace/Oil-Spill-Benz/datasets/raw/dv3-combined'
    
    # Validate paths
    if not os.path.isdir(labels_source_directory):
        print(f"Error: Labels source directory not found: {labels_source_directory}")
        exit(1)
    
    if not os.path.isdir(images_source_directory):
        print(f"Error: Images source directory not found: {images_source_directory}")
        exit(1)
    
    # Run the organization
    organize_yolo_dataset(labels_source_directory, images_source_directory, output_directory)