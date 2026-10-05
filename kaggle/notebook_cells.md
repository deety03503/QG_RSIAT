# Kaggle notebook cells

Run these in a GPU-enabled Kaggle Notebook after attaching the required dataset(s).

```python
!python kaggle/bootstrap.py --dataset imageneta --data_root /kaggle/input
```

```python
!python -m pip install -q -r kaggle/requirements-kaggle.txt
```

```python
!python scripts/doctor.py --data_root /kaggle/input --dataset imageneta
```

```python
!python scripts/launch.py --mode single --config exps/adapter_imageneta.json \
    --data_root /kaggle/input --out /kaggle/working/outputs
```

```python
!python scripts/launch.py --queue configs/experiments/ablation_core.json \
    --data_root /kaggle/input --out /kaggle/working/outputs --time_budget_h 10
```

```python
!python scripts/collect_results.py --out /kaggle/working/outputs
```

For offline pretrained initialization, attach a checkpoint with a recognizable
ViT-B/16-IN21K filename or provide `--pretrained_weights /kaggle/input/.../weights.pth`.
The bootstrap requirements intentionally do not include PyTorch or torchvision.
