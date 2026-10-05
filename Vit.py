import os
from huggingface_hub import snapshot_download

# Tạo thư mục chứa model
local_dir = "./vit-base-patch16-224-in21k"
os.makedirs(local_dir, exist_ok=True)

# Tải các file cấu hình và trọng số PyTorch (bỏ qua các file của thư viện khác như JAX/TF để nhẹ hơn)
snapshot_download(
    repo_id="google/vit-base-patch16-224-in21k",
    local_dir=local_dir,
    ignore_patterns=["*.msgpack", "*.h5"]  # Bỏ qua Flax và TensorFlow weights
)

print(f"Đã tải xong! Các file nằm tại: {os.path.abspath(local_dir)}")