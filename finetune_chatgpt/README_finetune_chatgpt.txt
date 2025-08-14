Fine-Tuning GPT-4o 

This directory contains all the scripts and data used to fine-tune `gpt-4o-2024-08-06` on a MISR dataset medical images. The pipeline handles data preparation, splitting, conversion, fine-tuning, and job monitoring.


 Pipeline Overview

1. split_data.py
- Purpose: Randomly selects 1220 images (1/4th of images in RQ2/image_letters) from the complete dataset for fine-tuning.
- Output: Copies selected images to a `test_data` folder and logs filenames in `fine_tuning_images.txt`. The test_data folder is used for finetuning.

2. split_qa_letters.py
- Purpose: Filters the original QA JSON (`qa_letters.json`) to exclude the 1220 finetuning images.
- Output: Saves the remaining QA entries to `qa_letters_inference.json` for use as test data with all_experiments_chatgpt.py.
- Includes: Verification to ensure all referenced image files are present.

3. generate_jsonl.py
- Purpose: Converts the finetuning subset of QA data into OpenAI’s fine-tuning `.jsonl` format.
- Input: `qa_letters.json` and `fine_tuning_images.txt`.
- Output: `vision_train_1220.jsonl`.

4. finetune_chatgpt.py
- Purpose: Uploads the `.jsonl` file and launches a fine-tuning job using the OpenAI API.
- Key Steps:
  - Uploads training data.
  - Starts fine-tuning on `gpt-4o-2024-08-06`.
  - Monitors job ID and status.


5. status.py
- Purpose: Retrieves and displays the status of the running or completed fine-tuning job using the job ID.

File Structure

finetune_chatgpt/
├── fine_tuning_images.txt      # List of selected finetuning image filenames
├── test_data/                  # Folder containing selected images and copied JSON for finetuning
├── vision_train_1220.jsonl     # Final file used for OpenAI fine-tuning
├── split_data.py               # Image selection script
├── split_qa_letters.py         # Filtering QA JSON
├── generate_jsonl.py           # Conversion to .jsonl
├── finetune_chatgpt.py         # Runs fine-tuning job
├── status.py                   # Monitors fine-tuning job status

Final Output

After running the complete pipeline, you will:
- Have a fine-tuned GPT-4o model tailored to our dataset.
- Be able to compare performance before vs. after fine-tuning using separate evaluation scripts.


Fine tuned model : ft:gpt-4o-2024-08-06:viscom::Bp0Y8FmM

This model can only be accessed by our own OpenAI's API. You can export the API in Terminal using export OPENAI_API_KEY='the key'
