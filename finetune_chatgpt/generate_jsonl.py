import os
import json
import base64
from typing import List, Dict

def load_metadata(metadata_path: str) -> List[Dict]:
    """
    Loads metadata from a JSON file.

    Parameters:
    - metadata_path (str): Path to the JSON file containing metadata.

    Returns:
    - List[Dict]: List of metadata entries, each as a dictionary.
    """
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} metadata entries.")
    return data

def find_image_entries(metadata: List[Dict], image_dir: str) -> List[Dict]:
    """
    Finds image entries in metadata that match files in the image directory.

    Parameters:
    - metadata (List[Dict]): List of metadata entries.
    - image_dir (str): Directory containing image files.

    Returns:
    - List[Dict]: List of metadata entries that have matching image files.
    """
    print(f"Scanning image directory {image_dir} for PNG files...")
    available = set(os.listdir(image_dir))
    filtered = [entry for entry in metadata if entry['filename'] in available]
    print(f"Found {len(filtered)} matching images in metadata.")
    if not filtered:
        raise ValueError(f"No metadata entries matched files in {image_dir}")
    return filtered

def encode_image_to_data_url(image_path: str) -> str:
    """
    Encodes an image file to a data URL.

    Parameters:
    - image_path (str): Path to the image file to encode.

    Returns:
    - str: Data URL representing the encoded image.
    """
    with open(image_path, 'rb') as img_f:
        raw = img_f.read()
    b64_str = base64.b64encode(raw).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def make_example(entry: Dict, image_data_url: str, system_prompt: str) -> Dict:
    """
    Creates a single example for the JSONL file.

    Parameters:
    - entry (Dict): Metadata entry containing image information and question-answer pairs.
    - image_data_url (str): Data URL of the image.
    - system_prompt (str): System prompt to be included in the example.

    Returns:
    - Dict: Example in the format expected for the JSONL file.
    """
    qa = entry['question_answer'][0]
    question = qa['question']
    answer = "Yes" if qa['answer'] == 1 else "No"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]},
            {"role": "assistant", "content": answer}
        ]
    }

def write_jsonl(examples: List[Dict], output_path: str):
    """
    Writes a list of examples to a JSONL file.

    Parameters:
    - examples (List[Dict]): List of examples to write to the file.
    - output_path (str): Path to the output JSONL file.
    """
    print(f"Writing {len(examples)} examples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for ex in examples:
            out_f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print("✅ Write complete.")

def main(
    image_dir: str,
    metadata_path: str,
    output_path: str,
    system_prompt: str
):
    """
    Main function to generate JSONL file from image and metadata.

    Parameters:
    - image_dir (str): Directory containing image files.
    - metadata_path (str): Path to the metadata JSON file.
    - output_path (str): Path to the output JSONL file.
    - system_prompt (str): System prompt to be included in the examples.
    """
    all_meta = load_metadata(metadata_path)
    entries = find_image_entries(all_meta, image_dir)
    print(entries)

    examples = []
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        filename = entry['filename']
        print(f"[{idx}/{total}] Processing image: {filename}")
        img_path = os.path.join(image_dir, filename)

        image_data_url = encode_image_to_data_url(img_path)
        ex = make_example(entry, image_data_url, system_prompt)
        examples.append(ex)

    write_jsonl(examples, output_path)

if __name__ == "__main__":
    IMAGE_DIR     = "/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt/test_data"
    META_PATH     = "/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt/test_data/qa_letters.json"
    OUTPUT_PATH   = "/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt/vision_train_1220.jsonl"
    SYSTEM_PROMPT = (
        "You are an assistant that answers positional queries about organs "
        "in MISR images. Answer only Yes or No."
    )

    main(IMAGE_DIR, META_PATH, OUTPUT_PATH, SYSTEM_PROMPT)
