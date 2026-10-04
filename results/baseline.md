# RSIAT baseline reproduction

## Status

**Phase 2 is not complete.** The checkout contains one historical full-run log
for each of IN-R B0I20 and IN-A B0I20, but the required fresh three-seed
reproduction could not run in this session. The existing results are recorded
below as historical references only; they are not presented as newly verified
or as three-seed statistics.

## Historical results found in the repository

| Dataset/config | Run logged | `Average Accuracy (CNN)` (`Ā`) | Final task accuracy (`A_B`) | Reference in outline |
|---|---:|---:|---:|---:|
| IN-R B0I20 | seed 1993; CUDA device 0 | 86.924 | 82.75 | 86.92 / 82.75 |
| IN-A B0I20 | seed 1993; CUDA device 0 | 74.891 | 66.23 | 74.89 / 66.23 |

The logged values match the approximate figures in the outline. Each is a
single run. Mean ± standard deviation over three seeds is **not available**.

Raw historical logs:

- [IN-R log](../logs/adapter/imagenetr/0/20/all_1993_pretrained_vit_b16_224_in21k_adapter.log)
- [IN-A log](../logs/adapter/imageneta/0/20/all_1993_pretrained_vit_b16_224_in21k_adapter.log)

The logs are timestamped 2025-11-20. They contain the dataset config values,
CUDA device index and logged seed, but do not record the source commit hash,
runtime package versions, GPU model/VRAM, system RAM, or dataset/backbone
checksums. Therefore they do not establish that the current checkout and a
known runtime produced these results.

## Why a fresh reproduction could not run here

- The current workspace has no `data/datasets/` contents for IN-R/IN-A.
- The selected local Python environment is CPython 3.14.5 and PyTorch is not
  installed in it.
- `nvidia-smi` is unavailable and no Kaggle/CUDA/NVIDIA runtime variables were
  present in the local environment.
- No local ViT checkpoint was found. The model factory requests pretrained
  weights through `timm` (`utils/inc_net.py:9-54`), so a run without a mounted
  model/cache could attempt an unintended download. No dependencies were
  installed and no data or weights were downloaded.

On 2026-10-04, after the user requested skipping the smoke test and proceeding
to phase 2, both requested commands were attempted:

```text
python main.py --config ./exps/adapter_imagenetr.json
python main.py --config ./exps/adapter_imageneta.json
```

Both terminated during import with `ModuleNotFoundError: No module named
'torch'`. The IN-R/IN-A dataset directories and local Torch/Hugging Face model
caches were also absent. No training started and no new metric was produced.
The smoke test was not run, as requested.

## Seed and protocol caveat

The outline requires at least three training seeds while preserving class
order from seed 1993. Current code cannot provide that protocol as configured:

- `trainer.py:108-115` hard-codes Python, Torch and CUDA RNG seeds to `1`,
  regardless of the configured run seed.
- `data/data_manager.py:138-164` uses the configured seed to shuffle class
  order when `shuffle` is true.
- The experiment JSONs currently specify a one-element `seed: [1993]`.

Therefore simply varying JSON `seed` would vary class order but not the
explicit Torch/Python training seed, violating the required separation.
Do not count repeated runs of this unchanged code as independent seed trials.
The seed/order plumbing must be resolved before the required three-seed
baseline can be run; record any preparatory change separately and verify the
seed-1993 class order stays fixed.

## Required next evidence

Run the baseline on the specified Kaggle environment after confirming mounted
IN-R/IN-A data and ViT-B/16-IN21K weights are readable. Record Python,
PyTorch, CUDA runtime, driver, GPU model/count and per-GPU VRAM, system RAM,
dataset/weight paths, seed and commit for each run. Complete three runs per
dataset and preserve raw logs/configs before using these baselines as E0.
