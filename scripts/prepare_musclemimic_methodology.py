"""Download only pinned public source files; no models, motions or weights."""
from pathlib import Path
import urllib.request

from myo.vertical_mimic import COMMIT, SOURCE_FILES, source_manifest
import json


def main():
    root = Path("data/musclemimic_methodology")
    root.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/amathislab/musclemimic/{COMMIT}/{name}", timeout=60).read()
        target.write_bytes(data)
    (root/"PINNED_COMMIT").write_text(COMMIT+'\n')
    manifest = source_manifest(root)
    expected = json.loads(Path("configs/vertical_mimic_sources.json").read_text())
    if manifest != expected:
        raise ValueError("Downloaded MuscleMimic source does not match pinned hashes")
    print("Verified pinned public sources:", COMMIT)


if __name__ == "__main__":
    main()
