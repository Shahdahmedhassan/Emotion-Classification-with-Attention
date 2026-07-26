"""
Run this ONCE (locally or in the Kaggle notebook) to upload the trained
DistilBERT model to Hugging Face Hub as a MODEL repo (not a Space).
Model repos have no ZeroGPU quota limits, so this works on a free account.

Usage:
    pip install huggingface_hub
    python push_model_to_hub.py
"""

from huggingface_hub import HfApi, login

login()  # paste your HF token when prompted (needs "write" access)

api = HfApi()
username = api.whoami()["name"]

# Change this if you want a different model repo name
MODEL_REPO_ID = f"{username}/emotion-classifier-distilbert"

api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True, private=False)

# Path to the extracted "distilbert_emotion_model" folder from artifacts.zip
api.upload_folder(
    folder_path="artifacts/distilbert_emotion_model",
    path_in_repo=".",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
)

print(f"Model uploaded: https://huggingface.co/{MODEL_REPO_ID}")
print(f"Now set HF_MODEL_ID = \"{MODEL_REPO_ID}\" inside app.py")
