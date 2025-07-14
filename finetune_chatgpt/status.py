import openai
import time


job_id = "ftjob-xfz3CTKKQFzLMpLvELxOJZap" 


start_time = time.time()

print(f"🔍 Tracking fine-tuning job: {job_id}")
last_status = None

while True:
    job = openai.fine_tuning.jobs.retrieve(job_id)
    status = job.status
    if status != last_status:
        print(f"🟡 Status: {status}")
        last_status = status

    if status in ["succeeded", "failed", "cancelled"]:
        break

    time.sleep(10)  # wait 10 seconds before checking again

end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

print("\n📊 Final Report:")
print(f"Status: {job.status}")
print(f"Model: {job.fine_tuned_model}")
print(f"Training file: {job.training_file}")
print(f"Error: {job.error}")
print(f"⏱️ Time taken: {int(mins)} min {int(secs)} sec")

'''
🔍 Tracking fine-tuning job: ftjob-xfz3CTKKQFzLMpLvELxOJZap
🟡 Status: running
🟡 Status: succeeded

📊 Final Report:
Status: succeeded
Model: ft:gpt-4o-2024-08-06:viscom::Bp0Y8FmM
Training file: file-UdVNt7fM9Dxs1Vr3jrc1TR
Error: Error(code=None, message=None, param=None)
⏱️ Time taken: 57 min 46 sec
'''