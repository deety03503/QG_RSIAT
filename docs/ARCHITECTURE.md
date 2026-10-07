# QR-RSIAT / Kaggle architecture

## Scope and execution boundary

This change keeps RSIAT as the default method and adds QR-RSIAT as an explicit
configuration choice. The local machine is used for static inspection only.
Training is intentionally rejected when no CUDA device is available; use
`scripts/doctor.py` for CPU-side diagnostics and run training on a configured
Kaggle GPU session.

## Runtime flow

1. `main.py` loads the selected legacy experiment JSON, merges validated
   QR-RSIAT options, detects hardware, builds a `RuntimePlan`, and creates one
   `RuntimeContext`.
2. The main process writes `hardware.json`, `plan.json`, and `config.json`.
   It resolves the dataset (and compatible legacy symlink) before the learner
   initializes. Errors are re-raised after a traceback is written to
   `FAILED.txt` when possible.
3. `trainer.py` sets deterministic seeds, creates a per-seed checkpoint
   manager, constructs the original `DataManager` and learner factory, and
   resumes only from a checkpoint with matching configuration and dataset
   metadata.
4. `models/RSIAT_adapter.py` creates the classifier head and a fresh
   `StepModule` at each task boundary. The step wrapper owns the current
   network, enabled trainable losses/aligners/kernels, and the frozen prior
   network. In DDP, only this wrapper is distributed.
5. At task end, class means/covariances are merged from float64 sufficient
   statistics. Optional classifier compensation runs on rank 0 and broadcasts
   classifier state. The learner evaluates RSIAT inference, updates histories,
   and saves an atomic task checkpoint.

## Package responsibilities

| Package / file | Responsibility |
|---|---|
| `qrsiat/config/` | Typed QR-RSIAT options, compatibility merge, early validation |
| `qrsiat/hardware/` | Hardware profile, runtime plan, callback OOM batch probe |
| `qrsiat/runtime/` | Device/precision context, deterministic seed, safe optimization fallbacks |
| `qrsiat/data/` | Dataset path validation/symlink, DataLoader policy, offline weight lookup |
| `qrsiat/distributed/` | Process-group setup, collectives and non-padding evaluation sampler |
| `qrsiat/stats/` | Float64 class sufficient statistics and globally merged drift |
| `qrsiat/quantum/` | Real-statevector circuit, feature map, kernels, aligner and losses |
| `qrsiat/training/` | DDP forward boundary, schedule, checkpoint and evaluation helpers |
| `qrsiat/runner/`, `scripts/` | Direct/queue launch, Kaggle doctor and result collection |
| `kaggle/`, `configs/experiments/` | Kaggle bootstrap guidance and opt-in ablation configurations |

The legacy model factory remains the registration point. `data/data.py` remains
the dataset implementation; the registry resolves its expected paths instead
of replacing its transforms or class-order behavior.

## Method and precision behavior

- Default `aligner="rae"` preserves the legacy RAE route. `qhybrid` is opt-in.
- Quantum modules are constructed only when their related options are enabled.
  Their simulator, fidelity kernels, and qhybrid calculations are explicitly
  run in full precision, outside autocast.
- The statevector is real-valued: RY/CNOT prepare the state and Pauli-X/Z
  expectation values are read out (two scalars per qubit). Pauli-Y is omitted
  because its expectation is zero for real-valued states.
- Inference and evaluation use the RSIAT network/classifier and do not invoke
  the quantum aligner or kernels.
- The frozen previous-task network is evaluated under `torch.inference_mode()`.
- `use_compile`, `freeze_cast`, `grad_checkpointing`, and `probe` are
  independent opt-ins (probe defaults on Kaggle). Compilation and frozen
  parameter casting fall back to eager/original dtype if setup or execution
  fails. Unsupported gradient-checkpointing hooks are logged and left disabled.
- `--smoke` limits a run to at most two tasks, epochs, and data batches per
  training/evaluation/statistics pass. It writes under an additional `smoke/`
  directory and is for wiring/throughput checks only; its metrics are not
  comparable to RSIAT results.
- `--time_budget_h` is queue-only: the scheduler stops dispatching new jobs
  near the configured reserve. Direct training rejects this option rather than
  claiming it can interrupt safely inside a task.
- The optional `uint8_cache` data backend is not implemented; no augmentation
  semantics are changed by a hidden cache path.

## Distributed behavior

- `auto` selects CPU diagnostics, one GPU, job-parallel queue execution, or
  homogeneous-NCCL DDP according to hardware and queue context.
- Job-parallel execution isolates one process per visible GPU. DDP is launched
  through `torch.distributed.run`; each rank gets a rank-specific loader seed
  and calls `DistributedSampler.set_epoch`.
- Device-specific primitives are confined to their runtime owners: only
  `RuntimeContext` constructs the selected CUDA device; only the DDP wrapper
  supplies `device_ids`; only the job-parallel queue sets
  `CUDA_VISIBLE_DEVICES`. These are dynamic dispatch points, not hard-coded
  device selections.
- DDP wraps `StepModule` once per task. Trainable components must contribute to
  that forward graph; the old frozen network is the only external model branch.
- Pairwise losses gather differentiable features when `pair_gather` is enabled.
  It defaults on for DDP. Without it, pairwise objectives are rank-local.
- The qrel student kernel is trainable, while its target is produced by a
  frozen copy snapshotted at the start of each incremental task. This keeps
  the relational target fixed while that task's student kernel is optimized.
- Class statistics and drift merge sums/counts with all-reduce. Evaluation
  deliberately runs the complete, unpadded test set on each rank; each rank
  reports the same metric and only rank 0 persists artifacts.
- Classifier compensation is rank-0-only, followed by collective barriers and
  broadcast of classifier state. Checkpoint writes and persistent result/log
  writes are rank-0-only. Dataset resolution/symlink creation runs on rank 0
  and its success or traceback is broadcast before any rank proceeds.
  Output-directory creation is idempotent across ranks.

## Data, weights, and artifacts

- ImageFolder data must expose matching `train/<class>` and `test/<class>`
  directories and the expected original class count. Kaggle data is resolved
  from `/kaggle/input` and linked into `data/datasets/` only when the target
  path is free.
- CIFAR-100 retains torchvision's dataset loader and configurable root via
  `RSIAT_CIFAR_ROOT`/`--data_root`; an attached/pre-cached archive is required
  when the Kaggle session has no network access.
- Pretrained ViT weights are selected from an explicit path, matching Kaggle
  input files, or timm's pretrained cache/download path. A supplied checkpoint
  must meet the adapter loader's compatible-backbone threshold; random
  initialization is not a silent fallback.
- Results are written under `<output_dir>/<dataset>/seed_<seed>/`. Per-task
  checkpoint files are atomic, hash-checked, and retain the newest two. Resume
  validates task schedule, class order, class count, and prototype shape.
- Kaggle dependencies intentionally exclude `torch` and `torchvision`.

## Source files touched for integration

- `main.py`, `trainer.py`
- `models/RSIAT_adapter.py`, `models/base.py`
- `data/data.py`
- `utils/inc_net.py`, `network/vision_transformer_adapter.py`
- `.gitignore`

## Verification boundary

No test suite, model, OOM probe, training run, or baseline reproduction was run
on the local machine. Static diagnostics and source review do not prove Kaggle
package compatibility, checkpoint compatibility, symlink permissions,
dataset layout, GPU memory capacity, or NCCL behavior. See
[`RISK_REGISTER.md`](RISK_REGISTER.md) before the first Kaggle run.
