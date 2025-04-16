from huggingface_hub import HfFileSystem, hf_hub_download
import argparse

def fetch_checkpoint(repo_name, checkpoint_folder, local_checkpoint_folder):
    # To resume training, you need to get the previous optimizer state,
    # use the following code to download the checkpoint files locally
    # checkpoint_folder = "global_step1000"
    # local_checkpoint_folder = "data/OpenR1-Qwen-7B-nsa-B1024-hwfalse-1000"

    # List the files in the checkpoint folder
    fs = HfFileSystem()
    files_in_folder = fs.ls(f"{repo_name}/{checkpoint_folder}", detail=False)
    files_in_folder = [file.split("/")[-1] for file in files_in_folder]

    # Download the files from the subfolder
    downloaded_files = []
    for file_name in files_in_folder:
        file_path = hf_hub_download(
            repo_id=repo_name,
            filename=file_name,
            subfolder=checkpoint_folder,
            local_dir=local_checkpoint_folder,
        )
        downloaded_files.append(file_path)
        
if __name__ == "__main__":
    # fetch_checkpoint(
    #     "ZMC2019/OpenR1-Qwen-7B-nsa-B1024-hwfalse",
    #     "global_step1000",
    #     "data/OpenR1-Qwen-7B-nsa-B1024-hwfalse-1000"
    # )
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--local_checkpoint_folder", type=str, required=True)
    args = parser.parse_args()
    fetch_checkpoint(args.repo_name, args.checkpoint_folder, args.local_checkpoint_folder)