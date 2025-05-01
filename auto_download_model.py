# auto_download_model.py
import os
import requests
from tqdm import tqdm

MODEL_NAME = "ggml-gpt4all-j-v1.3-groovy.bin"
MODEL_URL  = f"https://gpt4all.io/models/{MODEL_NAME}"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
dest_path = os.path.join(MODEL_DIR, MODEL_NAME)

if not os.path.exists(dest_path):
    print(f"Downloading {MODEL_NAME} (~3GB) to {dest_path}")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('Content-Length', 0))
        with open(dest_path, 'wb') as f, tqdm(
                desc=MODEL_NAME,
                total=total,
                unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print("Download complete!")
else:
    print(f"Model already exists at {dest_path}")
