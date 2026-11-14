#!/usr/bin/env python3
"""Fetch the reviewed public BodyParts3D arm subset; keep all assets in data/."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

BASE = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/"
LOCK = Path(__file__).resolve().parents[1] / "integrations/unreal/bodyparts3d-source.json"


class ArchiveRanges(io.RawIOBase):
    """Read selected ZIP members without downloading the entire 143 MB archive."""

    def __init__(self, lock):
        self.url, self.size, self.etag = lock["url"], lock["archive_bytes"], lock["etag"]
        self.position = self.transferred = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size - self.position, self.size - self.position)
        if size == 0:
            return b""
        if not 0 < size < 4 * 1024**2:
            raise ValueError("Unexpected archive range size")
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.position}-{self.position + size - 1}", "If-Match": self.etag})
        with urllib.request.urlopen(request, timeout=30) as response:
            expected = f"bytes {self.position}-{self.position + size - 1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Official server did not return the pinned archive range")
            payload = response.read(size + 1)
        if len(payload) != size:
            raise RuntimeError("Incomplete archive range")
        self.position += size
        self.transferred += size
        return payload


def checked(payload, expected, name):
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"BodyParts3D checksum mismatch: {name}")
    return payload


def prepare(output):
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    # Record rights and individual asset identities before importing any mesh.
    (output / "preparation-plan.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    for entry in lock["catalogs"]:
        target = source / entry["file"]
        if target.exists():
            checked(target.read_bytes(), entry["sha256"], target.name)
        else:
            with urllib.request.urlopen(BASE + entry["file"], timeout=30) as response:
                target.write_bytes(checked(response.read(), entry["sha256"], target.name))
    missing = [entry for entry in lock["assets"] if not (source / (entry["element_id"] + ".obj")).exists()]
    if missing:
        with ArchiveRanges(lock) as remote, zipfile.ZipFile(remote) as archive:
            for entry in missing:
                payload = checked(archive.read(entry["member"]), entry["sha256"], entry["element_id"])
                (source / (entry["element_id"] + ".obj")).write_bytes(payload)
                print(f"Downloaded {entry['myo_id']}: {entry['element_id']}", flush=True)
    catalog = (source / "isa_element_parts.txt").read_text(encoding="utf-8")
    for entry in lock["assets"]:
        checked((source / (entry["element_id"] + ".obj")).read_bytes(), entry["sha256"], entry["element_id"])
        if f"{entry['fma_id']}\t{entry['name']}\t{entry['element_id']}" not in catalog:
            raise ValueError(f"Catalog identity mismatch: {entry['myo_id']}")
    manifest = lock | {"prepared_utc": datetime.now(timezone.utc).isoformat(), "modifications": [],
                       "scope": "Unmodified source surfaces; no fit, active muscle law, or Flesh asset yet."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verified {len(lock['assets'])} BodyParts3D surfaces in {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/bodyparts3d"))
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
