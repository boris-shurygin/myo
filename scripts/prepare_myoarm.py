#!/usr/bin/env python3
"""Download the reviewed public MyoArm revision and prepare the local 1-DOF arm."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

from myo.geometry import MYOARM_COMMIT, MYOARM_MUSCLES, prepare_model

REPOSITORY = "https://github.com/MyoHub/myo_sim"
RAW = f"https://raw.githubusercontent.com/MyoHub/myo_sim/{MYOARM_COMMIT}"


def fetch(path: str, destination: Path, expected: str) -> dict[str, str]:
    target = destination / "upstream" / path
    if target.exists():
        payload = target.read_bytes()
    else:
        with urllib.request.urlopen(f"{RAW}/{path}", timeout=60) as response:
            payload = response.read()
    digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if digest != expected:
        raise ValueError(f"Source checksum mismatch for {path}; remove the changed local file to retry.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {"path": path, "url": f"{RAW}/{path}", "git_blob_sha1": digest,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "license": "Apache-2.0 upstream; MoBL-ARMS noncommercial / OpenSim CC-BY-3.0 provenance; see docs/myoarm.md"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/myoarm"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Record the reviewed source/rights before downloading model or mesh files.
    manifest = {"repository": REPOSITORY, "commit": MYOARM_COMMIT,
                "usage": "local noncommercial personal/research prototype",
                "license_review": "docs/myoarm.md",
                "license_sources": [f"{REPOSITORY}/blob/{MYOARM_COMMIT}/LICENSE",
                    "https://simtk.org/projects/upexdyn/",
                    "https://github.com/amathislab/musclemimic_models/blob/main/NOTICE"],
                "muscle_names": list(MYOARM_MUSCLES),
                "files": [{"path": path, "url": f"{RAW}/{path}", "git_blob_sha1": digest,
                           "license": "Apache-2.0 upstream; MoBL-ARMS noncommercial / OpenSim CC-BY-3.0 provenance; see docs/myoarm.md"}
                          for path, digest in SOURCE_BLOBS.items()]}
    manifest_path = args.output / "manifest.json"
    # A failed re-download must not erase the last successful asset manifest.
    # Keep the pre-download per-file rights/provenance record separately.
    (args.output / "preparation-plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=4) as executor:
        manifest["files"] = list(executor.map(
            lambda entry: fetch(entry[0], args.output, entry[1]), SOURCE_BLOBS.items()))
    prepare_model(args.output)
    for path in ("model.xml", "muscles.json"):
        manifest.setdefault("derived", []).append({
            "path": path, "sha256": hashlib.sha256((args.output / path).read_bytes()).hexdigest()})
    manifest["modifications"] = [
        "Only elbow flexion remains mobile; remaining joints fixed at upstream q=0",
        "Six elbow paths retained with original attachment and wrap geometry",
        "All native actuators removed; external delayed series-elastic force path only",
        "Fixed world mounting, elbow lower limit 15 degrees with no upper limit, hand-fixed payload, camera",
        "Noncontact bone rendering; no skin or volumetric muscle surface created",
    ]
    staged_manifest = args.output / "manifest.json.tmp"
    staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    staged_manifest.replace(manifest_path)
    print(f"Prepared {args.output / 'model.xml'} ({len(manifest['files'])} verified source files)")


# Immutable source hashes from the reviewed commit's Git tree; not a moving ref.
SOURCE_BLOBS: dict[str, str] = {
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
    "myo_sim/models/arm/README.md": "87e3a494e3da2f74bcb49f03410aa228baa333e9",
    "myo_sim/models/arm/assets/myoarm_r_assets.xml": "9eccf8fcc72da87610a46d3eed545b9a9a579e15",
    "myo_sim/models/arm/assets/myoarm_r_chain.xml": "102aa4811c2b798ac07c52b78087acdc929b08cc",
    "myo_sim/models/arm/assets/myoarm_r_muscle.xml": "aa23580f3b07044bd315d42b6b82bec5adf60397",
    "myo_sim/models/arm/assets/myoarm_r_tendon.xml": "636963d7a5f654f0c1b56c4ef369825c6cd233a0",
    "myo_sim/models/meshes/1mc.stl": "bed700289d2f893105df7d92ad09ed8ad1ce200a",
    "myo_sim/models/meshes/2distph.stl": "dc0dd30f54020259c6b99746459c407159c41bbc",
    "myo_sim/models/meshes/2mc.stl": "1b3aecab3f6fe9afd8d7034c100e038072227084",
    "myo_sim/models/meshes/2midph.stl": "04a5a25910bfc46d3cf031f730433a3e2536609b",
    "myo_sim/models/meshes/2proxph.stl": "c07e375c7d640b128e69faa447f6f7032a6a37f3",
    "myo_sim/models/meshes/3distph.stl": "1c07524034a336d9795753cad5e14c937f8e570d",
    "myo_sim/models/meshes/3mc.stl": "6293415f95d1d652c67b82cbe5266882106c459c",
    "myo_sim/models/meshes/3midph.stl": "19bdb430e2f3be304274dee50e08a07d20b3b913",
    "myo_sim/models/meshes/3proxph.stl": "de216d406c740a408ea3cd1a780b221df6615a32",
    "myo_sim/models/meshes/4distph.stl": "f3083ca44100d5b0f70ea8843c936adb02f124d9",
    "myo_sim/models/meshes/4mc.stl": "22829871fd28f7e2ea8e498ec48b01ad55ea16c3",
    "myo_sim/models/meshes/4midph.stl": "b3a7d1a9ee326b67a44d7a821a1230d6fc3356c5",
    "myo_sim/models/meshes/4proxph.stl": "ca5ee3567d0c6207e83f8c31b528204bd19e428c",
    "myo_sim/models/meshes/5distph.stl": "96c695990ff12104eeb02572dd38b7669cbde31d",
    "myo_sim/models/meshes/5mc.stl": "31dc18ad96a6c2d59165a98bec2f9dfa1d2b3a90",
    "myo_sim/models/meshes/5midph.stl": "057f89588b215868551ea0ba6eb16dbe9fca6cfa",
    "myo_sim/models/meshes/5proxph.stl": "e2709b22e28ddc014daec166b0e44efea3529fde",
    "myo_sim/models/meshes/capitate.stl": "f97531474d837814ed46074e8f8283d8afdf5ea9",
    "myo_sim/models/meshes/clavicle.stl": "e3248f7a81cb877bd282604f315872d92952cb48",
    "myo_sim/models/meshes/hamate.stl": "71c79e8fdcbc5f7e11502b538b7f8719c0b13e0a",
    "myo_sim/models/meshes/humerus.stl": "dfb0fac5badf01cb35642a529e3b5a18d80f737f",
    "myo_sim/models/meshes/lunate.stl": "e5f085aa7f948bf7333024827d74d296f796966c",
    "myo_sim/models/meshes/pisiform.stl": "a6484d922de00a848ccfe0d5fe0aa204a6b58c19",
    "myo_sim/models/meshes/radius.stl": "b5d8f2616c3277df014a879d785aa1dd6b56c198",
    "myo_sim/models/meshes/scaphoid.stl": "708b95e00b9743a7df3ace66f6051a3d132fc3d7",
    "myo_sim/models/meshes/scapula.stl": "930bd22e1ee20ce667cb2bb28d1bb7defbc74564",
    "myo_sim/models/meshes/thorax.stl": "d4242ff7ccf4cbd91f479a93275f78a75237a938",
    "myo_sim/models/meshes/thumbdist.stl": "a44eddbfe18a1f47589282f3271e2b9e90cd3465",
    "myo_sim/models/meshes/thumbprox.stl": "54d45582e77184591910f5bfdbbff7c08e9c8058",
    "myo_sim/models/meshes/trapezium.stl": "49e02af096bd5482de4560201c184940125703f8",
    "myo_sim/models/meshes/trapezoid.stl": "e6cebda9192aa564927a746b01fb5d67cb503f38",
    "myo_sim/models/meshes/triquetrum.stl": "ea2eb19f2a4b5f392af9aa87b044f109f9fee3ad",
    "myo_sim/models/meshes/ulna.stl": "6bee343d44bb581e9bf88b6d764661e2f7bc58b0"
}


if __name__ == "__main__":
    main()
