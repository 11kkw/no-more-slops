"""
Download the data this experiment needs (idempotent). Environment is handled by uv via pyproject.toml.

    uv run scripts/setup_data.py            # COCO val2017 + annotations + DCSM checkpoints
    uv run scripts/setup_data.py --no-ckpt  # skip the Google Drive checkpoints
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
COCO_IMG = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DCSM_DRIVE = "https://drive.google.com/drive/folders/1HngVf3BixeRnpa0kEuCx5nucGFrQ3Dv6"


def fetch_zip(url, dest_dir, marker):
    if os.path.exists(os.path.join(dest_dir, marker)):
        print("exists:", marker)
        return
    os.makedirs(dest_dir, exist_ok=True)
    zpath = os.path.join(dest_dir, os.path.basename(url))
    print("downloading", url)
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest_dir)
    os.remove(zpath)
    print("done:", marker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ckpt", action="store_true")
    args = ap.parse_args()
    coco = os.path.join(DATA, "coco")
    fetch_zip(COCO_IMG, coco, "val2017")
    fetch_zip(COCO_ANN, coco, os.path.join("annotations", "instances_val2017.json"))
    if not args.no_ckpt:
        ck = os.path.join(DATA, "dcsm_ckpt")
        if os.path.exists(os.path.join(ck, "dcsm_obja_ckpt.pt")):
            print("exists: dcsm checkpoints")
        else:
            import gdown
            os.makedirs(ck, exist_ok=True)
            gdown.download_folder(DCSM_DRIVE, output=ck, quiet=False)
            sub = os.path.join(ck, "dcsm_checkpoints")
            if os.path.isdir(sub):
                for f in os.listdir(sub):
                    shutil.move(os.path.join(sub, f), os.path.join(ck, f))
                os.rmdir(sub)
            print("done: dcsm checkpoints")
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())


if __name__ == "__main__":
    main()
