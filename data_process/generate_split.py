import os
import json
import random
import argparse
from tqdm import tqdm

def generate_split(processed_dir, pc_dir, output_path):
    """
    Scans for .pkl files, checks for corresponding .ply files, and generates
    a train/val/test split file.
    """
    print(f"Scanning for .pkl files in: {processed_dir}")
    
    valid_items = []
    
    if not os.path.isdir(processed_dir):
        print(f"Error: Directory not found at {processed_dir}")
        return

    filenames = [f for f in os.listdir(processed_dir) if f.endswith('.pkl')]
    
    for filename in tqdm(filenames, desc="Verifying file pairs"):
        if not filename.endswith('.pkl'):
            continue
            
        parts = filename.replace('.pkl', '').split('_')
        if len(parts) != 2:
            continue
            
        sub_folder, file_no = parts
        
        # Check for corresponding point cloud file
        pc_filename = f"{file_no}.ply"
        pc_filepath = os.path.join(pc_dir, sub_folder, pc_filename)
        
        pkl_filepath = os.path.join(processed_dir, filename)

        condition_path = os.path.join('/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/conditions', sub_folder, file_no)

        if os.path.exists(pc_filepath) and os.path.exists(pkl_filepath) and os.path.exists(condition_path):
            valid_items.append(f"{sub_folder}/{file_no}")

        

    print(f"Found {len(valid_items)} valid items with both .pkl and .ply files.")

    # Shuffle the data
    random.shuffle(valid_items)

    # Split the data (80% train, 10% validation, 10% test)
    total_count = len(valid_items)
    train_split = int(total_count * 0.99)
    val_split = int(total_count * 0.995)

    train_data = valid_items[:train_split]
    val_data = valid_items[train_split:val_split]
    test_data = valid_items[val_split:]

    split_data = {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

    # Write to output file
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=4)

    print(f"Successfully generated split file at: {output_path}")
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate train/val/test split from processed data.")
    parser.add_argument('--processed_dir', type=str, default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/processed/cad_step',
                        help='Directory containing the .pkl files.')
    parser.add_argument('--pc_dir', type=str, default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/pc_cad',
                        help='Directory containing the point cloud .ply files.')
    parser.add_argument('--cond_dir', type=str, default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/conditions',
                        help='Directory containing the point cloud .ply files.')
    parser.add_argument('--output_path', type=str, default='/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/filtered.json',
                        help='Path to save the output JSON split file.')
    
    args = parser.parse_args()
    
    generate_split(args.processed_dir, args.pc_dir, args.output_path)