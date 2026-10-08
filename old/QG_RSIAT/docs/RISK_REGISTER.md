# First Kaggle run risk register

Static review only; likelihood and impact are qualitative. Resolve the
high-priority items during the first GPU-enabled Kaggle session before starting
a long experiment.

| Priority | Risk | Likelihood / impact | Detection | Mitigation / safe fallback |
|---|---|---|---|---|
| P1 | Pretrained ViT checkpoint missing, mismatched, or unavailable offline | Medium / Critical | Run `scripts/doctor.py` with the intended model weight path; model initialization verifies compatible backbone coverage | Attach a matching checkpoint and pass `--pretrained_weights`; otherwise use timm's supported cache/network path. Loader fails rather than silently using random backbone weights. |
| P1 | Kaggle dataset folder layout, class names/order, or expected class count differs from RSIAT | Medium / Critical | `scripts/doctor.py --dataset ... --data_root /kaggle/input` and startup registry validation | Attach the original split with `train/` and `test/`; inspect class mapping before training. Symlink creation refuses to replace a conflicting local path. |
| P1 | Kaggle notebook disk/runtime budget is insufficient for task checkpoints or a full run | Medium / High | Inspect free disk/hardware report and monitor output size/time | Use `--time_budget_h` with `scripts/launch.py --queue` to stop dispatching new jobs, run small batches, preserve outputs between sessions, and resume from attached output with `--resume_from`. Direct training refuses this option; it cannot be interrupted safely inside a task. |
| P1 | GPU memory is below the configured workload, including incremental QR branches | Medium / High | Kaggle-default CUDA probe records selected batch in `plan.json`; inspect CUDA OOM logs | Probe uses synthetic forward/backward and a safety margin. Lower batch size or enable `--grad_checkpointing`/`--freeze_cast` explicitly if needed. DDP probing is skipped to avoid asymmetric collective stalls. |
| P2 | Frozen-parameter casting consumes additional host RAM to preserve a rollback copy | Medium / Medium | Observe host RAM when `--freeze_cast` is enabled | The option is off by default. Disable it if host RAM is constrained; cast setup/forward failure restores the original parameter dtype. |
| P1 | Full per-class covariance and float64 sufficient-statistic buffers consume substantial memory and checkpoint space | Medium / High | Monitor peak GPU/RAM/disk while computing class statistics and saving each task | Checkpoint covariance is stored as a CPU float32 tensor (not expanded Python lists); use a Kaggle GPU/RAM/disk tier that can hold the full covariance workload. Low-memory covariance approximations are not enabled because they change classifier-compensation behavior. |
| P1 | Checkpoint resume is rejected or restores an incompatible run | Medium / High | `latest.json`, configuration digest and learner metadata are checked before task iteration | Keep config, dataset class order, task schedule, and output together. Use `--force_resume` only when intentional; schedule/order/shape checks still apply. |
| P1 | A checkpoint copied from an untrusted source may contain unsafe pickle payloads | Low / Critical | Resume loads Python/PyTorch RNG state as part of the checkpoint payload | Resume only checkpoints produced by a trusted RSIAT run. Pointer paths are constrained to the selected checkpoint directory; this does not make arbitrary checkpoint contents safe. |
| P2 | Kaggle mount does not permit symlink creation in the repository | Low / High | Startup raises with the exact source and target paths | Place data under the expected `data/datasets/<name>` path or enable symlink support in the runtime. Existing conflicting paths are never overwritten. |
| P2 | NCCL initialization or peer-to-peer transport fails | Medium / High | `scripts/doctor.py` reports NCCL; DDP setup logs retry/fallback outcome | Prefer job-parallel queue mode for independent ablations. If NCCL setup fails, use single-GPU mode; DDP is not silently emulated. |
| P2 | Pairwise loss semantics change under DDP or micro-batch accumulation | Medium / High | Inspect `plan.json` warnings and `pair_gather` in `config.json` | Keep differentiable pair gathering enabled for DDP. Avoid accumulation where possible; use a global batch divisible by world size. Rank-local pair mode is explicit and changes the objective. |
| P2 | DataLoader worker count, storage throughput, or pinned-memory pressure hurts runtime stability | Medium / Medium | Hardware report and epoch throughput/worker errors | Tune `--num_workers` and `--max_workers`; worker count zero disables prefetch/persistent workers. No `uint8_cache` path is enabled. |
| P2 | DDP evaluation/checkpoint/control-flow divergence causes a collective stall | Low / Critical | Compare per-rank logs and last completed task; keep an external session timeout | All ranks follow task collectives; evaluation uses the full unpadded test split on each rank; only rank 0 writes files. Prefer single GPU if diagnosing a hang. |
| P2 | The implemented real-valued RY/CNOT state preparation and Pauli-X/Z readout may differ from the paper's exact gate order or feature-map equations | Medium / High | Compare simulator operations, bit ordering, and X/Z expectation values with the method specification | The readout uses X/Z observables; Pauli-Y is omitted because its expectation is zero for real-valued states. Confirm the detailed circuit equations before interpreting quantum ablations. |
| P2 | Smoke-run metrics are mistaken for complete experiment results | Medium / Medium | Check `config.json` and the `smoke/` output path | Smoke mode caps tasks, epochs, and batches; use it only to check wiring. Its outputs are isolated and are not comparable to full runs. |
| P3 | Torch/timm/torchvision versions differ from the statically inspected environment | Medium / High | `hardware.json` records versions; import/initialization errors surface early | Start from Kaggle's preinstalled Torch stack. Install only `kaggle/requirements-kaggle.txt`; it deliberately excludes Torch and torchvision. |

## Not validated locally

- CUDA execution, AMP stability, peak memory, OOM fallback and DDP/NCCL behavior.
- Actual Kaggle dataset mounts, class mappings, image decoding and filesystem
  symlink policy.
- Whether the chosen offline checkpoint is the expected ViT-B/16 source and
  whether timm's cache is populated.
- Numerical behavior of the real-valued statevector simulator and exact
  equivalence to the method's full gate-order/feature-map specification.
- Full task-boundary resume on Kaggle, including hardware/device-count changes.
- Training accuracy or comparison with any RSIAT baseline.
