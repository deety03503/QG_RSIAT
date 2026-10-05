from pathlib import Path


checkpoint_path = (
    Path(__file__).resolve().parent
    / "vit-base-patch16-224-in21k"
    / "pytorch_model.bin"
)
if not checkpoint_path.is_file():
    raise FileNotFoundError(
        "Local ViT-IN21K checkpoint was not found: {}".format(checkpoint_path)
    )

print("Using local ViT-IN21K checkpoint: {}".format(checkpoint_path))
