#!/usr/bin/env python3
import os
import urllib.request
import sys
from pathlib import Path

MODEL_DIR = Path("/home/marcio/.cache/huggingface/hub/models--unsloth--Qwen2.5-VL-7B-Instruct-GGUF/snapshots/68bb8bc4b7df5289c143aaec0ab477a7d4051aab")
OUT_FILE = MODEL_DIR / "mmproj-F16.gguf"
URL = "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-F16.gguf"

if OUT_FILE.exists():
    print(f"Removing corrupted partial file ({OUT_FILE.stat().st_size} bytes)...")
    OUT_FILE.unlink()

print(f"Downloading complete mmproj-F16.gguf from Hugging Face...")

def reporthook(blocknum, blocksize, totalsize):
    read = blocknum * blocksize
    if totalsize > 0:
        pct = read * 100 / totalsize
        mb_read = read / (1024 * 1024)
        mb_total = totalsize / (1024 * 1024)
        sys.stdout.write(f"\rProgress: {pct:.1f}% ({mb_read:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()

urllib.request.urlretrieve(URL, OUT_FILE, reporthook)
print(f"\nDone! Verified file size: {OUT_FILE.stat().st_size / (1024 * 1024):.1f} MB")
