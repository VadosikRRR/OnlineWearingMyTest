"""Optional Kaggle/helper download: only three weight files, never repositories."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    args.output.mkdir(parents=True, exist_ok=True)
    sources = (("lygitdata/garmentiq", "hrnet.pth", "hrnet.pth"),
               ("lllyasviel/Annotators", "body_pose_model.pth", "body_pose_model.pth"),
               ("yisol/IDM-VTON", "humanparsing/parsing_lip.onnx", "parsing_lip.onnx"))
    downloaded = []
    for repository, filename, local_name in sources:
        revision = HfApi().model_info(repository).sha
        cached = hf_hub_download(repo_id=repository, filename=filename, revision=revision)
        destination = args.output / local_name
        shutil.copy2(cached, destination)
        downloaded.append(dict(repository=repository, filename=filename, revision=revision, local_name=local_name))
        print(destination)
    (args.output / "sources.json").write_text(json.dumps(downloaded, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
