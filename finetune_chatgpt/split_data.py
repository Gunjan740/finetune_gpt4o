import os
import random
import shutil

def select_and_copy_images(
    source_dir='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/MISR_Dataset/RQ2/images_letters',
    dest_dir='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt/test_data',
    log_dir='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt',
    json_file='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/MISR_Dataset/RQ2/qa_letters.json',
    num_images=1220,
    seed=42,
    log_file='fine_tuning_images.txt'
):
    """
    Randomly selects a specified number of images from a source folder,
    copies them to a destination folder, and logs their filenames.

    Also copies a reference JSON file (e.g., qa_letters.json) to the same folder
    for consistency in fine-tuning or testing workflows.

    Parameters:
    - source_dir (str): Folder containing the full set of images.
    - dest_dir (str): Destination folder for copied images.
    - log_dir (str): Folder where the log file will be saved.
    - json_file (str): Path to the original QA JSON file to be copied alongside images.
    - num_images (int): Number of images to randomly select and copy.
    - seed (int): Seed for reproducible random selection.
    - log_file (str): Filename to store the list of selected images.
    """
    # Set seed for reproducibility
    random.seed(seed)

    # Ensure destination and log directories exist
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Clear destination folder before copying
    for filename in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Get all image files (excluding non-images)
    image_files = [
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    if num_images > len(image_files):
        raise ValueError("Not enough images in the source directory.")

    # Randomly select images
    selected_images = random.sample(image_files, num_images)

    # Copy selected images
    for img in selected_images:
        shutil.copy(os.path.join(source_dir, img), os.path.join(dest_dir, img))

    # Copy the qa_letters.json file
    if os.path.exists(json_file):
        shutil.copy(json_file, os.path.join(dest_dir, os.path.basename(json_file)))
    else:
        print("Warning: JSON file not found at the specified path.")

    # Write copied image names to log file (excluding JSON)
    log_path = os.path.join(log_dir, log_file)
    with open(log_path, 'w') as log:
        for img in selected_images:
            log.write(img + '\n')

    print(f"{num_images} images copied to {dest_dir}. JSON copied. Log saved at {log_path}.")



def backup_and_delete_selected_images(
    source_dir='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/MISR_Dataset/RQ2/images_letters',
    backup_dir='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/MISR_Dataset/RQ2/images_letters_backup',
    log_file='/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt/fine_tuning_images.txt'
):
    """
    Backs up the source image folder and deletes selected images listed in a log file.

    Parameters:
    - source_dir (str): Folder containing the original image dataset.
    - backup_dir (str): Directory where the backup of source_dir will be saved.
    - log_file (str): File containing names of images to be deleted (one per line).
    
    Steps:
    1. Creates a backup of the entire image folder.
    2. Deletes only the images listed in the log file from the source folder.
    """
    import shutil

    # Step 1: Backup the source directory
    if not os.path.exists(backup_dir):
        shutil.copytree(source_dir, backup_dir)
        print(f"Backup created at: {backup_dir}")
    else:
        print(f"Backup directory already exists at: {backup_dir}. Skipping backup.")

    # Step 2: Read the list of images to delete
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    with open(log_file, 'r') as f:
        images_to_delete = [line.strip() for line in f.readlines()]

    # Step 3: Delete images from source folder
    deleted = 0
    for img in images_to_delete:
        img_path = os.path.join(source_dir, img)
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted += 1

    print(f"Deleted {deleted} images from source directory using log: {log_file}")


if __name__ == "__main__":
    select_and_copy_images()
    #backup_and_delete_selected_images()