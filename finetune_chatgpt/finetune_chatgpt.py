import openai
import os


# === STEP 2: Set paths and model ===
jsonl_path = "/home/gunjan/CascadeProjects/advanced_visual_deep_learning/finetune_chatgpt/vision_train_1220.jsonl"
model_name = "gpt-4o-2024-08-06"

# === STEP 3: Upload file ===
print("Uploading training file...")
with open(jsonl_path, "rb") as f:
    upload_response = openai.files.create(file=f, purpose="fine-tune")
training_file_id = upload_response.id
print(f"✅ Uploaded. File ID: {training_file_id}")

# === STEP 4: Start fine-tuning ===
print("Starting fine-tuning job...")
job_response = openai.fine_tuning.jobs.create(
    training_file=training_file_id,
    model=model_name,
    hyperparameters={
        "n_epochs": 3,
        #"learning_rate_multiplier": 0.1,
        #"lora_r": 8,
        #"lora_alpha": 16,
        #"lora_dropout": 0.1
    }

)
job_id = job_response.id
print(f"🎯 Fine-tuning started! Job ID: {job_id}")

# === STEP 5: Print job details ===
print("Job details:")
print(f"• Status: {job_response.status}")
print(f"• Created: {job_response.created_at}")
print(f"• Base model: {job_response.model}")
print(f"• Fine-tune model: {job_response.fine_tuned_model}")

print("\n📡 You can monitor this job using:")
print(f"openai fine_tuning.jobs.retrieve -i {job_id}")
