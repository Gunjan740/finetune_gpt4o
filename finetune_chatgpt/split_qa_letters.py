import json
import os

def create_qa_test_json_with_verification(
    qa_json_path='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/MISR_Dataset/RQ2/qa_letters.json',
    image_list_path='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt/fine_tuning_images.txt',
    image_folder='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/MISR_Dataset/RQ2/images_letters',
    image_key='filename'
):      
    """
    Creates a filtered QA JSON file for testing by removing entries that were used for finetuning,
    and verifies that all referenced image files exist in the specified image folder.

    Parameters:
    - qa_json_path (str): Path to the full QA dataset (original JSON file).
    - image_list_path (str): Path to the text file containing image filenames used for finetuning.
    - image_folder (str): Path to the folder containing all image files.
    - image_key (str): The key in each JSON entry used to match image filenames (default: 'filename').

    Steps:
    1. Loads the original QA JSON and the list of images used for finetuning.
    2. Filters out entries in the QA JSON that match the finetuning images.
    3. Saves the remaining entries into a new JSON file named 'qa_letters_inference.json'
       in the same directory as the original.
    4. Verifies that all image filenames in the new JSON exist in the image folder.
    5. Prints a summary report, including missing or extra image files.

    Output:
    - A filtered JSON file ready for inference/testing.
    - Terminal output showing counts and validation results.
"""
    # Load QA JSON
    with open(qa_json_path, 'r') as f:
        qa_data = json.load(f)
    original_count = len(qa_data)

    # Load copied image names
    with open(image_list_path, 'r') as f:
        images_to_exclude = set(line.strip() for line in f if line.strip())

    # Filter entries
    filtered_data = [entry for entry in qa_data if entry.get(image_key) not in images_to_exclude]
    filtered_count = len(filtered_data)
    removed_count = original_count - filtered_data.__len__()

    # Generate output path in same directory as original QA JSON
    output_path = os.path.join(os.path.dirname(qa_json_path), 'qa_letters_inference.json')

    # Save filtered entries
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    # Verification 1: Files in images_letters folder
    image_folder_files = set(f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    # Verification 2: Filenames referenced in new JSON
    json_filenames = set(entry.get(image_key) for entry in filtered_data)

    # Cross-check JSON vs folder
    missing_files = json_filenames - image_folder_files
    extra_files = image_folder_files - json_filenames

    # Report
    print("✅ === QA JSON Filtering Report ===")
    print(f"Original JSON entries       : {original_count}")
    print(f"Entries removed (for train) : {removed_count}")
    print(f"Remaining entries (for test): {filtered_count}")
    print(f"Test JSON saved to          : {output_path}")
    print()
    print("📁 === Image File Verification ===")
    print(f"Total images in folder      : {len(image_folder_files)}")
    print(f"Images referenced in JSON   : {len(json_filenames)}")
    print(f"✅ Matched files             : {len(json_filenames & image_folder_files)}")
    
    if missing_files:
        print(f"⚠️ Missing files in folder not found for JSON ({len(missing_files)}):")
        for f in list(missing_files)[:10]:
            print(f"   - {f}")
    else:
        print("✅ All JSON image references are present in the image folder.")

    if extra_files:
        print(f"ℹ️ Extra files in image folder not referenced in JSON ({len(extra_files)}).")

if __name__ == "__main__":
    create_qa_test_json_with_verification()
