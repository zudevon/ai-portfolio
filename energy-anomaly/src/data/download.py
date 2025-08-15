#!/usr/bin/env python3
from __future__ import annotations
import io
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm
import yaml

def _stream_download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}") as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def download_and_extract(config_path: str = "config.yaml") -> Path:
    """Download UCI power consumption zip and extract the .txt into data/raw/"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    url = cfg["data"]["download_url"]
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "household_power_consumption.zip"
    if not zip_path.exists():
        _stream_download(url, zip_path)
    else:
        print(f"Found existing {zip_path}, skipping download.")
    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        target = [m for m in members if m.endswith(".txt")]
        if not target:
            raise RuntimeError("No .txt file found in the zip archive.")
        txt_name = target[0]
        zf.extract(txt_name, raw_dir)
        print(f"Extracted {txt_name} into {raw_dir}")
        return raw_dir / txt_name

if __name__ == "__main__":
    download_and_extract()
