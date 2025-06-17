from openai import OpenAI

with open("keys.txt", "r") as f:
    org_id = f.readline().strip().split("=")[1]
    api_key = f.readline().strip().split("=")[1]

client = OpenAI(api_key=api_key, organization=org_id)
job_id = "ftjob-X8fJ3sZO82DNLRb8sGWiga57"  # Replace with your actual job ID

job = client.fine_tuning.jobs.retrieve(job_id)
print(job)

import json

for fname in [
    "dev/047_functions/finetune_01/047_func_01_train_oai.jsonl",
    "dev/047_functions/finetune_01/047_func_01_test_oai.jsonl",
]:
    print(f"Checking {fname}")
    with open(fname, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line)
            except Exception as e:
                print(f"Error in {fname} on line {i}: {e}")

from openai import OpenAI

client = OpenAI(api_key=api_key, organization=org_id)
file = client.files.retrieve("file-BDvhp9WsHoDyLKTQMPJRWK")
print(file)
file = client.files.retrieve("file-JtqvdLgZXSqizY3KDizYfK")
print(file)
