import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def find_image_file(directory, base_name):
    """Find the corresponding image file in a directory (supports multiple extensions)."""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF', '.TIFF']:
        image_path = os.path.join(directory, base_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None

def convert_labelme_to_yolo_seg(input_dir, output_dir):
    """
    Convert LabelMe JSON files into the YOLO Segmentation format:
    generates .png masks and corresponding .txt files.
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting conversion: LabelMe JSON -> YOLO Segmentation format ---")
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")

    all_files = os.listdir(input_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    total_image_count = sum(1 for f in all_files if f.lower().endswith(tuple(image_extensions)))
    json_files = [f for f in all_files if f.endswith('.json')]
    total_json_count = len(json_files)
    
    success_count = 0
    failure_count = 0
    
    if not json_files:
        print(f"Error: No .json files found in folder '{input_dir}'.")
        return

    print(f"\n--- Scan results ---")
    print(f"Found {total_image_count} image files in the source folder.")
    print(f"Found {total_json_count} JSON files; preparing to process...")
    print("-" * 25)

    for json_filename in tqdm(json_files, desc="Processing progress"):
        json_path = os.path.join(input_dir, json_filename)
        base_name = os.path.splitext(json_filename)[0]

        # --- ★★★ New fix: handle filenames ending with "__1" ★★★ ---
        # If the base filename ends with "__1", strip it to match the original image filename
        if base_name.endswith('__1'):
            base_name = base_name[:-3]  # remove the last three characters
        # --- ★★★ End of modification ★★★ ---

        # Find the corresponding image file
        image_path = find_image_file(input_dir, base_name)
        if image_path is None:
            tqdm.write(f"Warning: corresponding image for '{json_filename}' not found (tried '{base_name}'); skipped.")
            failure_count += 1
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            mask = Image.new('L', (img_width, img_height), 0)
            draw = ImageDraw.Draw(mask)
            yolo_txt_lines = []

            for shape in data['shapes']:
                if shape['label'] == '0' or shape['label'] == 'oil':
                    polygon = shape['points']
                    
                    polygon_tuples = [tuple(p) for p in polygon]
                    draw.polygon(polygon_tuples, fill=255)

                    normalized_points = []
                    for x, y in polygon:
                        norm_x = x / img_width
                        norm_y = y / img_height
                        normalized_points.append(f"{norm_x:.6f} {norm_y:.6f}")
                    
                    yolo_line = "0 " + " ".join(normalized_points)
                    yolo_txt_lines.append(yolo_line)

            mask_output_path = os.path.join(output_dir, base_name + '.png')
            mask.save(mask_output_path)

            if yolo_txt_lines:
                txt_output_path = os.path.join(output_dir, base_name + '.txt')
                with open(txt_output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_txt_lines))
            
            success_count += 1

        except Exception as e:
            tqdm.write(f"Error: problem processing file '{json_filename}': {e}")
            failure_count += 1

    print("\n--- All files processed! ---")
    print("\n--- Processing summary ---")
    print(f"Source image count: {total_image_count}")
    print(f"Source JSON count: {total_json_count}")
    print("----------------------")
    print(f"✅ Successful conversions: {success_count}")
    print(f"❌ Failed or skipped: {failure_count}")
    print("----------------------")


if __name__ == "__main__":
    # --- User configuration block ---
    input_directory = r'/home/robby/workspace/Oil-Spill-Benz/datasets/raw/DV3/Categorize_UAV'
    output_base_directory = r'/home/robby/workspace/Oil-Spill-Benz/datasets/raw/dv3-all'

    if not os.path.isdir(input_directory):
        print(f"Error: source path '{input_directory}' is not a valid folder.")
    elif not output_base_directory:
        print("Error: output folder path cannot be empty.")
    else:
        # Get all subdirectories in input_directory
        subdirs = [d for d in os.listdir(input_directory) 
                   if os.path.isdir(os.path.join(input_directory, d))]
        
        if not subdirs:
            print(f"No subdirectories found in '{input_directory}'")
        else:
            print(f"\nFound {len(subdirs)} subdirectories to process:")
            for subdir in subdirs:
                print(f"  - {subdir}")
            print()
            
            # Process each subdirectory
            for subdir in subdirs:
                input_subdir = os.path.join(input_directory, subdir)
                output_subdir = os.path.join(output_base_directory, subdir)
                
                print(f"\n{'='*60}")
                print(f"Processing: {subdir}")
                print(f"{'='*60}")
                
                convert_labelme_to_yolo_seg(input_subdir, output_subdir)