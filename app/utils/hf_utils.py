import os
from huggingface_hub import hf_hub_download, login

def download_model(repo_id: str, filename: str, local_path: str) -> str:
    """
    Downloads the model from Hugging Face if it does not exist locally.
    If the repository is public, no token is required; if private, the token will be used.
    Returns the local path to the downloaded model.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # If the model already exists locally, skip the download
    if os.path.exists(local_path):
        print(f"✅ Model found locally: {local_path}")
        return local_path

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

    # Try to log in if a token is available
    if hf_token:
        try:
            login(token=hf_token)
        except Exception as e:
            print(f"⚠️ Could not log in to Hugging Face: {e}")

    print(f"⬇️ Downloading {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token
        )
        print(f"✅ Model downloaded at: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        raise
