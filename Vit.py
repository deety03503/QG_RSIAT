from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = Path(__file__).resolve().parent / "vit-base-patch16-224-in21k"

snapshot_download(
    repo_id="google/vit-base-patch16-224-in21k",
    local_dir=str(local_dir),
    allow_patterns=["pytorch_model.bin"],
)

print(f"Đã tải xong! Các file nằm tại: {local_dir}")