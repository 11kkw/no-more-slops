"""
Resumable chunked download of the DCSM training data from Google Drive.

Google intermittently answers plain downloads of these large public files with a 2 KB
"Too many users have viewed or downloaded this file recently" HTML page, but HTTP Range
requests keep working. So: fetch the size with a 1-byte range, then pull the file in
fixed-size ranges, verifying every chunk is binary and exactly the requested length.
Safe to re-run: resumes from the current file size.

    uv run scripts/drive_chunked_download.py            # both files
    uv run scripts/drive_chunked_download.py --only objaverse
"""
import argparse
import os
import sys
import time

import requests

FILES = {
    "dcsm": ("1cNMDrDO8DUziUPCJdIWpzBOQ1S9V8yYj", "dcsm_train_data_zipped.zip"),
    "objaverse": ("1gD4Cu5z_mhVGFZyzuUDkr4UctTBIZIXJ", "objaverse_train_cutmix_newrelations_withbg.zip"),
}
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
CHUNK = 256 * 1024 * 1024


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def total_size(sess, url):
    for attempt in range(30):
        r = sess.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        cr = r.headers.get("Content-Range", "")
        if r.status_code == 206 and "/" in cr:
            return int(cr.split("/")[-1])
        log(f"size probe got {r.status_code} {r.headers.get('Content-Type')}; retry in 60s")
        time.sleep(60)
    raise RuntimeError("could not get file size")


def fetch_range(sess, url, start, end):
    """Return bytes for [start, end] inclusive, or None on a quota page / short read."""
    r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120, stream=True)
    if r.status_code != 206 or "text/html" in r.headers.get("Content-Type", ""):
        return None
    buf = bytearray()
    for part in r.iter_content(1 << 20):
        buf += part
    return bytes(buf) if len(buf) == end - start + 1 else None


def download(key, out_dir):
    fid, name = FILES[key]
    url = URL.format(id=fid)
    path = os.path.join(out_dir, name)
    sess = requests.Session()
    size = total_size(sess, url)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    log(f"{name}: total {size / 1e9:.2f} GB, have {have / 1e9:.2f} GB")
    if have > size:
        raise RuntimeError(f"{path} is larger than the remote file; delete it")
    t0, done0 = time.time(), have
    with open(path, "ab") as f:
        pos = have
        fails = 0
        while pos < size:
            end = min(pos + CHUNK, size) - 1
            data = fetch_range(sess, url, pos, end)
            if data is None:
                fails += 1
                wait = min(120, 10 * fails)  # refusals are intermittent, not a hard lockout
                log(f"{name}: chunk {pos}-{end} refused (quota/short); sleeping {wait}s")
                time.sleep(wait)
                continue
            f.write(data)
            f.flush()
            pos = end + 1
            fails = 0
            rate = (pos - done0) / max(1e-6, time.time() - t0) / 1e6
            log(f"{name}: {pos / 1e9:.2f}/{size / 1e9:.2f} GB  {rate:.1f} MB/s")
    log(f"{name}: DONE {os.path.getsize(path)} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FILES), default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "data", "dcsm_train", "dcsm_training"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for key in ([args.only] if args.only else list(FILES)):
        download(key, args.out)
    log("ALL DONE")


if __name__ == "__main__":
    main()
