
import os
from concurrent.futures import ThreadPoolExecutor

from psutil import cpu_count
from huggingface_hub import hf_hub_download
import huggingface_hub as hfhub


def download(repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT"):
    token = os.getenv("HF_TOKEN")
    assert token is not None, "Please set the HF_TOKEN environment variable"
    os.makedirs("data", exist_ok=True)

    def _download(filename: str):
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir="data", repo_type="dataset", token=token)
            return 1, filename
        except hfhub.hf_api.HTTPError:
            return 0, filename

    files = hfhub.list_repo_files(repo_id, token=token, repo_type="dataset")

    # Make sure that the files are named correctly
    slash_train_in_files = any("train/" in file for file in files)
    slash_val_in_files = any("val/" in file for file in files)
    assert slash_train_in_files and slash_val_in_files, "Please download the data from the HuggingFace dataset page"

    # Only download the "bytes" files; they already contain the tokens
    files = [file for file in files if "bytes" in file]

    nproc = cpu_count() - 2  # leave some space for user
    print(f"Downloading {len(files)} files...")

    while True:
        with ThreadPoolExecutor(nproc) as executor:
            feedback = list(executor.map(_download, files))
        
        files = [file for success, file in feedback if success == 0]
        if not files:
            break
        print(f"Downloading {len(files)} files...")


if __name__ == "__main__":
    download()
    