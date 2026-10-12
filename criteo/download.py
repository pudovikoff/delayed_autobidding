"""Download the Criteo Attribution Modeling for Bidding log (Diemert et al., 2017).

The file comes from Criteo's mirror on the Hugging Face Hub, criteo/criteo-attribution-dataset
(licence CC BY-NC-SA 4.0), and is checked against the SHA-256 of the copy used in the paper.
"""
import hashlib
import os
import shutil
import sys
import urllib.request

URL = "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/criteo_attribution_dataset.tsv.gz"
SHA256 = "94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "criteo",
                    "criteo_attribution_dataset.tsv.gz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(DEST) and sha256(DEST) == SHA256:
        print("already downloaded:", DEST)
        return
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print("downloading 653 MB from", URL)
    with urllib.request.urlopen(URL) as r, open(DEST + ".part", "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if sha256(DEST + ".part") != SHA256:
        os.remove(DEST + ".part")
        sys.exit("checksum mismatch")
    os.replace(DEST + ".part", DEST)


if __name__ == "__main__":
    main()
