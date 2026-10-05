
# Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models

  

This repository serves as the official implementation corresponding to the paper titled "Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models". 
![Overall pipeline of RSIAT. ](images/framework.png)

## Installation
### Requirements
Ubuntu 20.04 LTS

Python 3.10

CUDA 11.8

Detailed package information and corresponding versions are available in the requirements.txt file.

### Data preparation

The overall directory structure should be:
```
RSIAT/
├──data/
├──datasets/
│   ├──cifar-100-python/
│   ├──cub/
│   ├──imagenet-a/
│   ├──imagenet-r/
│   ├──omnibenchmark/
│   ├──vtab/
│   ......
├──.......
```

## Training and evaluation

The training and evaluation instructions for each dataset are in the "./args.sh" file. Each dataset can be calculated separately, and the results are stored in the "./logs" folder.

## Kaggle GPU runs (QR-RSIAT integration)

The original RSIAT configuration remains the default. QR-RSIAT options are
validated by `main.py`; select `--aligner qhybrid` only for experiments that
explicitly enable its objectives. Local CPU-only environments can run
diagnostics, but training fails fast without CUDA.

On Kaggle, enable a GPU, attach the dataset with the original RSIAT
`train/<class>` and `test/<class>` layout, then run:

```bash
python kaggle/bootstrap.py --dataset imageneta --data_root /kaggle/input
python -m pip install -r kaggle/requirements-kaggle.txt
python scripts/doctor.py --data_root /kaggle/input --dataset imageneta
python scripts/launch.py --mode single --config exps/adapter_imageneta.json \
  --data_root /kaggle/input --out /kaggle/working/outputs
```

For a short wiring check, add `--smoke` (at most two tasks, epochs, and batches;
outputs go under an additional `smoke/` directory and are not research results).

Use `--pretrained_weights` to identify an attached offline ViT checkpoint when
needed. The Kaggle requirements intentionally do not replace the preinstalled
`torch` or `torchvision`. For queue-based ablations, resume, and notebook cell
examples, see [`kaggle/notebook_cells.md`](kaggle/notebook_cells.md), and review
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) before a long run.

## Citation

If you find this useful in your research, please consider citing:

```text
@inproceedings{zhao2026representation,
  title={Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models},
  author={Zhao, Jiarui and Huang, Libo and Li, Xiangqi and An, Zhulin and Yang, Chuanguang and Wang, Yu and Diao, Boyu and Xu, Yongjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18010--18020},
  year={2026}
}
```

## Acknowledgement

This repo is based on [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) and [SSIAT](https://github.com/HAIV-Lab/SSIAT).

Thanks for their wonderful work!!!
