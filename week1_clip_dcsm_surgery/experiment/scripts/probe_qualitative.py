"""
Qualitative heatmaps: for one image and a few text queries, render the per-patch cosine map
for each last-block readout (naive = DCSM code path, maskclip, clearclip) at the given sizes.
Usage:
  uv run scripts/probe_qualitative.py --image path.jpg --queries "car" "school bus" --out out/qual_bus.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPModel, CLIPTokenizer  # noqa: E402

from dcsm_surgery.clip_readouts import dense_features, heatmaps, preprocess, text_features  # noqa: E402
from dcsm_surgery.constants import MODES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 448])
    ap.add_argument("--readout", default="dcsm_tok", choices=["pooled", "dcsm_tok"])
    ap.add_argument("--prompt", default="a photo of a {}.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(args.model)
    patch = model.config.vision_config.patch_size
    txt_pooled, txt_tokens, content = text_features(model, tok, [args.prompt.format(q) for q in args.queries], device)

    img = Image.open(args.image).convert("RGB")
    rows = len(args.sizes) * len(args.queries)
    cols = 1 + len(MODES)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    r = 0
    for size in args.sizes:
        grid = size // patch
        px = preprocess(img, size).to(device)
        feats, _ = dense_features(model, px, size)
        for qi, q in enumerate(args.queries):
            axes[r, 0].imshow(img.resize((size, size)))
            axes[r, 0].set_title(f"{q} @ {size}px ({grid}x{grid})", fontsize=9)
            axes[r, 0].axis("off")
            for mi, m in enumerate(MODES):
                hm = heatmaps(feats[m][0, 1:], txt_pooled, txt_tokens, content, grid)[args.readout][qi].cpu().numpy()
                up = F.interpolate(torch.from_numpy(hm)[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
                ax = axes[r, 1 + mi]
                ax.imshow(img.resize((size, size)))
                ax.imshow(up, cmap="jet", alpha=0.55)
                ax.set_title(f"{m}  [{hm.min():.2f},{hm.max():.2f}]", fontsize=9)
                ax.axis("off")
            r += 1
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
