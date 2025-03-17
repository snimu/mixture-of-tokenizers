
import os
from concurrent.futures import ThreadPoolExecutor

from psutil import cpu_count
from huggingface_hub import hf_hub_download
import huggingface_hub as hfhub


def download(
        repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT",
        num_train_files: int = 700,  # TODO: update when true number is known
        num_fm_val_files: int = 1,
        num_fw_val_files: int = 1,
):
    token = os.getenv("HF_TOKEN")
    assert token is not None, "Please set the HF_TOKEN environment variable"

    def _download(filename: str):
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir="data", repo_type="dataset", token=token)
            return 1, filename
        except hfhub.hf_api.HTTPError:
            return 0, filename

    nproc = cpu_count() - 2  # leave some space for user
    files = [f"train_batch_{i+1}.bin" for i in range(num_train_files)]
    files += [f"val_batch_finemath_{i+1}.bin" for i in range(num_fm_val_files)]
    files += [f"val_batch_fineweb_{i+1}.bin" for i in range(num_fw_val_files)]
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
    