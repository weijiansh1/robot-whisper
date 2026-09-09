#!/usr/bin/env python3
"""Install and verify the pinned benchmark data without touching model services."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import shutil
import subprocess
import urllib.parse
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE.parent.parent / "himoe-vla-cache/libero-extensions"
PRO_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
PLUS_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
PLUS_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"


def digest(path, algorithm="sha256", git_blob=False):
    value = hashlib.new(algorithm)
    if git_blob:
        value.update(("blob %d\0" % path.stat().st_size).encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pro_files():
    tree_file = CACHE / "pro-hf-tree.json"
    subprocess.run([
        "curl", "-fLsS", "--connect-timeout", "15", "--max-time", "120", "--retry", "3",
        "https://hf-mirror.com/api/datasets/zhouxueyang/LIBERO-Pro/tree/"
        + PRO_REVISION + "?recursive=true&limit=1000", "-o", str(tree_file),
    ], check=True)
    entries = json.loads(tree_file.read_text())
    if len(entries) >= 1000:
        raise RuntimeError("The data tree requires pagination")
    files = [entry for entry in entries if entry["type"] == "file"]
    root = CACHE / "LIBERO-PRO/libero/libero"
    provenance = CACHE / "pro-dataset-metadata"
    provenance.mkdir(exist_ok=True)

    def fetch(entry):
        relative = pathlib.PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid dataset path")
        target = (root if relative.parts[0] in ("bddl_files", "init_files") else provenance) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        lfs = entry.get("lfs")
        expected = lfs["oid"] if lfs else entry["oid"]
        algorithm = "sha256" if lfs else "sha1"

        def valid(path):
            return (path.is_file() and path.stat().st_size == entry["size"]
                    and digest(path, algorithm, git_blob=not lfs) == expected)

        if not valid(target):
            temporary = target.with_name(target.name + ".download")
            method = "resolve" if lfs else "raw"
            url = "https://hf-mirror.com/datasets/zhouxueyang/LIBERO-Pro/" + method + "/" + PRO_REVISION
            url += "/" + urllib.parse.quote(str(relative))
            subprocess.run([
                "curl", "-fLsS", "--connect-timeout", "15", "--max-time", "120", "--retry", "3",
                url, "-o", str(temporary),
            ], check=True, capture_output=True)
            if not valid(temporary):
                raise RuntimeError("Dataset checksum mismatch: " + str(relative))
            temporary.replace(target)
        return {"path": str(relative), "bytes": entry["size"], "sha256": digest(target)}

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(fetch, files):
            records.append(record)
            if len(records) % 50 == 0:
                print("Pro verified %d/%d files" % (len(records), len(files)), flush=True)
    report = {"revision": PRO_REVISION, "verified_files": records, "bytes": sum(r["bytes"] for r in records)}
    (CACHE / "pro-dataset-install.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Pro dataset verified: %d files" % len(records), flush=True)


def plus_assets(archive):
    if archive.stat().st_size != 6395849578 or digest(archive) != PLUS_SHA256:
        raise RuntimeError("Plus assets.zip size/SHA-256 mismatch")
    root = CACHE / "LIBERO-plus/libero/libero"
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        required = sum(member.file_size for member in members)
        if shutil.disk_usage(root).free < required + 5 * 1024**3:
            raise RuntimeError("Insufficient space for Plus assets and 5 GiB reserve")
        destinations = []
        prefixes = set()
        for member in members:
            relative = pathlib.PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or relative.parts.count("assets") != 1:
                raise ValueError("Unexpected asset archive path: " + member.filename)
            asset_index = relative.parts.index("assets")
            prefixes.add(relative.parts[:asset_index])
            destinations.append(root.joinpath(*relative.parts[asset_index:]))
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected symlink in asset archive")
        if len(prefixes) != 1:
            raise ValueError("Inconsistent archive prefixes")
        print("Extracting Plus: %.2f GiB, %d entries" % (required / 1024**3, len(members)), flush=True)
        # Reading each entry during extraction also verifies its ZIP CRC.
        for member, target in zip(members, destinations):
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
    report = {
        "revision": PLUS_REVISION, "archive_sha256": PLUS_SHA256,
        "archive_bytes": archive.stat().st_size, "uncompressed_bytes": required,
        "entries": len(members), "zip_crc_verified": True,
        "stripped_archive_prefix": "/".join(next(iter(prefixes))),
    }
    (CACHE / "plus-assets-install.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pro", action="store_true")
    parser.add_argument("--plus-archive", type=pathlib.Path)
    args = parser.parse_args()
    if args.pro:
        pro_files()
    if args.plus_archive:
        plus_assets(args.plus_archive)
