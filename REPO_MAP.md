# REPO_MAP: RSIAT implementation review for QR-RSIAT

This map records the repository state before implementation. Source references
are file paths and line numbers from base commit
`c7550318d549dce6183e999bf1a46b8737bd604e`.

## Repository and run flow

- The repository origin is `https://github.com/deety03503/QG_RSIAT.git`.
  The current branch `qr-rsiat` was created from `main` at
  `c7550318d549dce6183e999bf1a46b8737bd604e` for this work. Existing local
  changes in `README.md`, `Vit.py`, `agents/`, and `docs/` were present before
  the branch switch and were not modified.
- The tracked application is a compact Python/PyTorch project: `main.py`,
  `trainer.py`, `models/`, `network/`, `utils/`, `data/`, `exps/`, and
  `args.sh`. There are six dataset JSON configs.
- `main.py:6-10` loads the selected JSON and calls `RSIAT_train`.
  `main.py:12-18` defines the sole CLI parameter, `--config`
  (default `./exps/adapter_imageneta.json`); there are no other CLI options.
  `main.py:26-28` merges with `vars(args).update(config)`, so JSON values
  override CLI values.
- `trainer.py:12-23` loops over `args["seed"]`, calls `_train`, and averages
  the returned task-average CNN top-1 values. `trainer.py:25-92` sets up
  logging, seeds, devices, `DataManager`, and the selected model, then calls
  `incremental_train`, `eval_task`, and `after_task` once per task, in that
  order. The evaluated score is logged before moving to the next task.
- All six `exps/adapter_*.json` files select `model_name: "adapter"` and
  `pretrained_vit_b16_224_in21k_adapter`. `utils/model_factory.py:1-7`
  dispatches this to `models/RSIAT_adapter.py:Learner`.
- `trainer.py:29-55` forms log paths and configures the file/console logger.
  It currently places results under `logs/`; the checkpoint save calls in
  `trainer.py:75-78` are commented out.
- `args.sh:1-6` invokes `main.py` once per each of six JSON files, passing only
  `--config`. The README says runs are separate and results go under `./logs`
  (`README.md:28-31`).

## Training, losses, prototypes, inference

| Concern | Finding |
|---|---|
| Classifier loss | `models/RSIAT_adapter.py:212-216` constructs `AngularPenaltySMLoss` with configured `scale` and `margin`; `utils/loss.py:5-42` implements CosFace (selected by the caller) and other angular losses. The learner passes new-class logits and targets offset by `_known_classes`. `network/classifier.py:9-37,51-58` normalizes feature vectors and classifier weights before returning logits. |
| Base-task representation-steering loss | `models/RSIAT_adapter.py:37-38,212-225,227-248`: `RS_Loss.forward(features, labels)` receives `[B,768]` features and `[B]` labels, normalizes features, forms `[B,B]` pairwise dot products, and applies positive `relu(1-dot)` / negative `relu(dot-margin)` terms. At task 0 it is weighted by `lambda_rs * min(1, epoch/warmup_epoch)` (`:218-221`). |
| Incremental alignment / orthogonality | `models/RSIAT_adapter.py:201-210`: `_inc_loss(features, features_old)` gets current and frozen-old `[B,768]` features. It transforms old features and prior class means through `old_ae`; MSE aligns current features to transformed old features. It normalizes those tensors, forms a `[num_old_classes,B]` similarity matrix, then averages all similarities. This is the original plain mean-similarity term, not a hinge/top-k quantum loss; weights are `beta` and `gamma`. |
| Projector / RAE | No symbol named RAE or projector was found. `utils/toolkit.py:84-102` defines `AutoencoderSigmoid`: encoder/decoder with hidden width 64, code width configured as `ae_code_dims`, a residual `reconstructed_x + x`, and sigmoid final decoder layer. `models/RSIAT_adapter.py:69-72` instantiates it at the start of task 1. It is not evidence of equivalence to an RAE Projector described elsewhere. |
| Previous-task features | `models/RSIAT_adapter.py:41-47` copies and freezes the previous `_network` after task evaluation. `models/RSIAT_adapter.py:223-224` extracts old features from that frozen copy for the next incremental task. |
| Prototypes/covariances and drift | `models/base.py:227-258` computes class feature means and covariance from current training examples. `models/RSIAT_adapter.py:105-118` computes displacement for old class means after incremental training and conditionally adds it when `ssca` is true. At a task's loss time, the loss sees means retained from previous tasks (including displacement applied after the preceding task if enabled); current-task displacement is applied only after training, so is not used by that same task's loss. |
| Classifier calibration | `models/base.py:52-142` samples 256 features/class from a multivariate Gaussian based on class means/covariances and updates classifier parameters; invoked by `models/RSIAT_adapter.py:116-118` if `ca` is true and `ca_epochs > 0`. |
| Inference/evaluation | `models/base.py:160-163,177-205` evaluates logits and top-k predictions from `self._network(inputs)`. `utils/inc_net.py:175-186` gets `[B,768]` ViT features, runs the classifier, and returns logits. Neither evaluation nor `forward` calls `old_ae`; preserving this path keeps the auxiliary training module out of inference. |
| Trainable parameters | `models/RSIAT_adapter.py:123-160` builds optimizer and cosine-annealing scheduler. Task 0 SGD groups selected ViT blocks and classifier; later-task SGD groups convnet, classifier, and `old_ae`. Later-task AdamW instead receives only `_network.parameters()` and excludes `old_ae`. Scheduler uses `CosineAnnealingLR(T_max=tuned_epochs, eta_min=min_lr)`. |

## Data, seeds, devices, parallelism

- Dataset order is created in `data/data_manager.py:138-164`. When
  `shuffle=true`, `np.random.seed(seed)` determines class order. This currently
  couples the configured training seed to class order. The configs set
  `"shuffle": true` for most datasets and `"seed": [1993]`; VTAB sets
  `"shuffle": false`.
- `trainer.py:108-115` ignores the requested seed and resets Python, Torch and
  CUDA RNGs to `1`. `DataLoader` shuffling has no explicit generator or worker
  initializer.
- Configured `seed` is a list (`exps/adapter_*.json`, key near the top of each
  file); `trainer.py:12-20` iterates it and overwrites `args["seed"]` with each
  scalar. `data/data_manager.py:138-164` then uses that same scalar to seed
  NumPy and shuffle class order when `shuffle` is true. This violates the
  outline's separation between training seeds and the fixed class-order seed
  1993.
- GPU setup is in `trainer.py:_set_device`. `num_worker` is the requested GPU
  count: `0` selects CPU, `1` selects `cuda:0`, and larger values select the
  first N visible GPUs. A request larger than the visible GPU count fails
  explicitly. GPU devices are stored internally as `torch.device` values.
- Train, test, and class-mean `DataLoader`s use the separate
  `data_loader_workers` setting (default `8`), not the GPU-count setting.
- Input batch size is dataset-configured (`batch_size` key in each
  `exps/adapter_*.json`) and is used for train/test loaders in
  `models/RSIAT_adapter.py:26,87-89`. Calibration creates a synthetic batch
  of 256 samples per class (`models/base.py:72-93`).
- Multi-GPU training wraps the current network and frozen old network in
  `nn.DataParallel`, splitting each forward pass across all visible GPUs.
  Classifier calibration wraps its classifier in `nn.DataParallel` while
  operating on synthetic feature batches.

## Configuration and dependencies

- `args.sh:1-6` launches one process per dataset, each specifying a JSON file.
- JSON configs contain per-dataset training settings, `seed: [1993]`, and
  `num_worker: 2`; launch each configured run with `python main.py`.
- `requirements.txt` pins `torch==2.8.0+cu126` and
  `torchvision==0.23.0+cu126`, while the modified `README.md` still describes
  Python 3.10 and CUDA 11.8. This mismatch is a compatibility risk; no package
  was installed and no runtime/baseline was attempted during this read-only
  mapping step.
- Image-folder dataset locations are hard-coded under `./data/datasets/` in
  `data/data.py`; the repository does not contain those datasets or Kaggle
  runtime mounts. This differs from the outline's generic `datasets/` layout.
- `Vit.py` downloads the `google/vit-base-patch16-224-in21k` checkpoint from
  Hugging Face. `utils/inc_net.py:9-54` creates the timm architecture with
  `pretrained=False` and loads weights from that local Hugging Face snapshot;
  it does not request pretrained weights from timm.
- Local checks on 2026-10-04: CPython 3.14.5 compiled 16 repository Python
  files in memory without syntax errors and parsed all six experiment JSON
  files. The selected `.venv` is Python 3.14.5 and has no installed `torch`;
  no PyTorch unit test or application import/training run could therefore be
  performed in this local environment. No test files were found in the
  repository.
- Existing tracked IN-R and IN-A logs report `Average Accuracy (CNN)` 86.924
  and 74.891 respectively, matching the reference values in the outline
  (`logs/adapter/imagenetr/0/20/all_1993_pretrained_vit_b16_224_in21k_adapter.log`,
  `logs/adapter/imageneta/0/20/all_1993_pretrained_vit_b16_224_in21k_adapter.log`).
  These are historical full runs (timestamps 2025-11-20), not a new two-task
  smoke test; they do not record full runtime/GPU/VRAM metadata and are not
  treated as proof that the current runtime or Kaggle inputs work.

## Internal dead code identified at the base commit

The following are candidates for the requested cleanup, based on repository-wide
search of the current Python files. This is an internal-reference audit, not
proof that external consumers import these APIs:

- `BaseLearner.displacement_cov` (`models/base.py:261-284`) has no caller;
  it invokes `self.cov_computation`, for which no definition was found in the
  repository.
- `DataManager.get_dataset_with_split` and `DataManager.getlen`
  (`data/data_manager.py:82-136,186-188`) have no in-repository callers.
- `BaseLearner._get_memory`, `exemplar_size`, and `samples_per_class`
  (`models/base.py:28-39,171-175`) have no in-repository callers.
- `utils.toolkit.target2onehot` (`utils/toolkit.py:27-30`) has no caller; its
  import in `models/RSIAT_adapter.py:14` is also unused.
- `utils.toolkit.makedirs` (`utils/toolkit.py:33-36`) has no caller.
- `SimpleVitNet.weight_align` (`utils/inc_net.py:188-203`) has no caller.
- `CosineLinear` (`network/classifier.py:61-88`) has no in-repository use;
  it is imported but not used by `utils/inc_net.py:5`. `reduce_proxies` is
  only used by that class.
- `utils/ops.py` has no in-repository imports. Its augmentation classes do
  not appear in any configured pipeline.

## Discrepancies and decisions recorded at mapping time

1. **Meaning of “remove unrelated code”:** the outline requires RSIAT-off
   behavior, original evaluation, baseline comparisons, classifier
   calibration, and multiple datasets, but does not list deletable files or
   legacy features. The cleanup below therefore removed only the enumerated
   in-repository dead code; all baseline, evaluation, calibration, and dataset
   paths were retained.
2. **Feature-map choice (confirmed):** use a trainable linear map from 768 to
   `q * lq` angles and subtract the per-batch angle mean. The outline's
   defaults remain `q=8`, `lq=2`.
3. **Quantum-kernel semantics:** the feature-map details (input scaling,
   centering source, angle mapping/readout) and the MLP kernel definition are
   not fully specified. These choices affect `qrel`, `qorth`, and fair
   parameter-count comparisons.
4. **Orthogonality threshold (policy confirmed):** use a fixed `epsilon`
   config value across tasks, not a per-task estimate. Its numeric value still
   needs to be selected using training/validation data only. The stated loss
   formula reduces by batch size and sums the selected top-k terms per sample.
5. **RAE equivalence:** the current `AutoencoderSigmoid` differs in name and
   visible structure from the outline's assumed RAE projector. The original
   RSIAT architecture/loss needs to be treated as the executable baseline
   unless the source paper or user confirms a replacement interpretation.
6. **Runtime-dependent milestones:** Kaggle datasets/weights, GPU count,
   VRAM, GPU smoke tests, multi-GPU validation, baselines and ablations are
   not available to verify from this local checkout. No results should be
   claimed until those runs happen on the specified runtime.
7. **Internal dead-code cleanup:** the user confirmed the existing baseline
   contains unused logic. The identified methods/helpers and `utils/ops.py`
   have since been removed from this checkout; external consumers of these
   previously unreferenced internals were not available for verification.

## Recommended safe scope

Keep original RSIAT training, evaluation, dataset definitions and calibration
available as the all-flags-off baseline. Implement QR-RSIAT as opt-in training
features and delete only code proven unreachable or redundant after checking
all callers. Leave pre-existing user changes untouched.

## Implementation status (working tree; no implementation commits)

- Added `quantum/` with a real PyTorch statevector simulator (RY gates and
  nearest-neighbor CNOT chain), trainable centered `768 -> q*lq` angle map,
  fidelity/cosine/RBF/MLP kernels, zero-initialized skip aligner, and qrel /
  top-k qorth losses. Readout is Pauli-Z only, as confirmed by the user; angle
  scaling is `pi * tanh(linear(features))`.
- For the classical qrel controls, the RBF kernel defaults to `gamma=1/d`; the
  MLP control is `tanh(cosine(x,y) + 1)`. These are explicit implementation
  choices and should be held fixed in ablations.
- Added CLI/config settings in `main.py` and all six experiment JSON files.
  JSON is the default source; explicitly provided CLI values override it.
  Baseline behavior remains the defaults (`aligner=rae`,
  `lambda_qrel=0`, `orth=plain`, `rs_kernel=cosine`). qorth uses fixed
  `orth_epsilon=0.5` as an initial config value; it still requires
  train/validation-based selection before experiments.
- Example opt-in run:

  ```bash
  python main.py --config ./exps/adapter_imagenetr.json --seed 1993 \
    --num_worker 1 --data_loader_workers 8 --aligner qhybrid --lambda_qrel 1.0 \
    --kernel quantum --orth qweighted
  ```

  `--rs_kernel quantum` enables the optional quantum base-task RS kernel.
  Classical qrel comparisons can select `cosine`, `rbf`, or `mlp`.
- Training RNG now follows the selected `seed`; class order is generated
  independently from `class_order_seed=1993`. Loader worker count and seeded
  generators are passed to train, test and class-mean DataLoaders.
- Multi-GPU training uses `nn.DataParallel`; the current network, frozen old
  network, and calibration classifier forward passes are distributed across
  the visible GPUs.
- Removed the dead-code candidates listed above, including the orphaned
  `utils/ops.py`; baseline classifier, task evaluation, data loading, drift,
  and Gaussian calibration code remain.
- No tests have been run, as requested. Kaggle execution, seed reproducibility,
  inference equivalence, and actual multi-GPU behavior remain unverified.
  Phase 2's three-seed baseline is still incomplete.
