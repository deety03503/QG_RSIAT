# AGENT_PLAN v2: Xây dựng kiến trúc repo QR-RSIAT, tự nhận diện phần cứng, chạy trên Kaggle

> Dành cho coding agent. Đây là bản kế hoạch **chỉ tập trung vào kiến trúc và viết mã**.
> Máy hiện tại **không có GPU** và **không chạy được mô hình**, nên:
> - **Không yêu cầu viết test, không yêu cầu chạy thử, không yêu cầu tái lập baseline** trên máy này.
> - Mã được chạy lần đầu trên Kaggle (clone repo từ GitHub). Vì vậy mã phải **phòng thủ**, **fail-fast với thông báo rõ ràng**, và có sẵn công cụ chẩn đoán chạy được trên Kaggle.
> - Repo gốc: https://github.com/zjrzjrz/RSIAT (MIT; kế thừa từ PILOT và SSIAT). Phương pháp: xem tài liệu QR-RSIAT (các cờ A–D ở mục 2).

---

## 0. Nguyên tắc chung

1. **Đọc tĩnh trước, viết sau.** Agent được đọc code gốc (không cần chạy) để lập `REPO_MAP.md`. Không viết dòng nào vào file gốc trước khi bản đồ xong.
2. **Không bao giờ hardcode thiết bị.** Không `.cuda()`, không `torch.device("cuda:0")`, không `nn.DataParallel`, không đường dẫn tuyệt đối. Mọi thứ đi qua một đối tượng `RuntimeContext`.
3. **Mặc định giữ nguyên hành vi RSIAT gốc.** Mọi tính năng mới nằm sau cờ cấu hình; cờ tắt thì logic huấn luyện không đổi.
4. **Thang tối ưu có dự phòng (optimization ladder).** Mỗi tối ưu rủi ro (compile, ép kiểu backbone, cache dữ liệu) bọc trong `try/except`, ghi log, và tự lùi về cấu hình an toàn nếu lỗi. Vì không thể test cục bộ, mặc định chọn phương án an toàn trước.
5. **Diff nhỏ vào file gốc.** Mã mới nằm trong package `qrsiat/`, `scripts/`, `kaggle/`, `configs/`. File gốc chỉ thêm các điểm gắn (hook) tối thiểu.
6. **Không cài lại torch.** Kaggle đã có torch/CUDA sẵn. File phụ thuộc cho Kaggle không được chứa `torch`, `torchvision`.
7. **Không bịa.** Chỗ nào chưa chắc về code gốc, ghi vào mục "Vấn đề mở" của `REPO_MAP.md` thay vì đoán.

---

## 1. Những gì đã biết về repo gốc

**Đã xác minh (từ trang chủ):** thư mục `data/`, `exps/`, `images/`, `logs/adapter/`, `models/`, `network/`, `utils/`; các file `main.py`, `trainer.py`, `args.sh`, `requirements.txt`; README yêu cầu Python 3.10, CUDA 11.8; dữ liệu đặt ở `datasets/<tên>/`; chạy bằng lệnh trong `args.sh`; kết quả vào `./logs`.

**Chưa biết, agent phải đọc để xác định:** tên file/class trong `models/`, `network/`, `utils/`; cấu trúc JSON trong `exps/`; nơi đặt cosine loss, $\mathcal{L}_{\mathrm{RS}}$, warm-up, RAE projector, $\mathcal{L}_{\mathrm{align}}$, $\mathcal{L}_{\mathrm{orth}}$, thu thập prototype/covariance, bù drift, retrain classifier, vòng eval.

---

## 2. Phạm vi tính năng (tóm tắt, chi tiết toán học nằm ở tài liệu QR-RSIAT)

| Mã | Tính năng | Cờ |
|---|---|---|
| A | Bộ căn chỉnh lai cổ điển–lượng tử, khởi tạo đồng nhất, thay RAE | `--aligner {rae,qhybrid}` |
| B | $\mathcal{L}_{\mathrm{qrel}}$: khớp ma trận kernel giữa $f^{t-1}$ và $f^{t}$ | `--lambda_qrel`, `--kernel {quantum,cosine,rbf,mlp}` |
| C | $\mathcal{L}_{\mathrm{orth}}^{q}$: top-$k$ + trọng số $\alpha$ + hinge | `--orth {plain,qweighted}` |
| D | (tùy chọn) kernel lượng tử trong $\mathcal{L}_{\mathrm{RS}}$ ở base task | `--rs_kernel {cosine,quantum}` |

Mạch lượng tử **chỉ dùng khi train**, bị bỏ khi suy luận. Chỉ dùng cổng RY và CNOT nên biên độ luôn thực, mô phỏng bằng tensor thực `[B, 2**q]`, tự viết bằng PyTorch (không PennyLane/Qiskit).

---

## 3. Kiến trúc thư mục đích

```
RSIAT/                              # repo gốc, giữ nguyên cấu trúc
├── main.py  trainer.py  args.sh    # chỉ thêm hook tối thiểu
├── models/  network/  utils/       # chỉ thêm hook tối thiểu
├── exps/                           # JSON gốc, giữ nguyên
│
├── qrsiat/                         # PACKAGE MỚI
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py               # dataclass cấu hình + validate (fail-fast)
│   │   └── merge.py                # hợp nhất args gốc (dict) với cờ mới
│   ├── hardware/
│   │   ├── detect.py               # HardwareProfile: đọc CPU/RAM/GPU/nền tảng
│   │   ├── plan.py                 # RuntimePlan: chọn mode, AMP, batch, workers...
│   │   ├── probe.py                # dò batch tối đa bằng OOM-probe lúc chạy
│   │   └── platform_kaggle.py      # nhận diện Kaggle, đường dẫn, giới hạn đĩa
│   ├── runtime/
│   │   ├── context.py              # RuntimeContext: device, rank, world, amp, scaler
│   │   ├── precision.py            # autocast/GradScaler/TF32 theo plan
│   │   └── optimize.py             # compile/ép kiểu backbone/checkpointing, có fallback
│   ├── distributed/
│   │   ├── setup.py                # init/cleanup process group, NCCL fallback
│   │   ├── collectives.py          # all_reduce_sum, gather_cat, broadcast_obj, barrier
│   │   └── samplers.py             # train sampler, eval sampler không padding
│   ├── data/
│   │   ├── registry.py             # tên dataset -> đường dẫn (Kaggle/local), symlink
│   │   ├── loaders.py              # DataLoader theo plan (workers, prefetch, pin)
│   │   └── weights.py              # tìm/đặt trọng số ViT-B/16-IN21K offline
│   ├── quantum/
│   │   ├── simulator.py            # statevector thực: RY, CNOT chain, đo Pauli
│   │   ├── feature_map.py          # 768 -> góc qubit, centering, chuẩn hóa
│   │   ├── kernels.py              # quantum/cosine/rbf/mlp cùng giao diện
│   │   ├── aligner.py              # QHybridAligner (skip + W_up=0 * Readout(PQC))
│   │   └── losses.py               # qrel_loss, qorth_loss
│   ├── stats/
│   │   ├── accumulators.py         # prototype/covariance cộng dồn, hiểu phân tán
│   │   └── drift.py                # ước lượng và bù drift, hiểu phân tán
│   ├── training/
│   │   ├── step_module.py          # StepModule.forward(batch) -> dict loss (bọc DDP)
│   │   ├── loop.py                 # vòng lặp epoch: accumulation, scaler, warm-up
│   │   ├── schedule.py             # warm-up beta(e), gamma(e), lambda_RS(e)
│   │   ├── checkpoint.py           # lưu/khôi phục theo task, ghi nguyên tử
│   │   └── eval.py                 # đánh giá phân tán, gom đúng số mẫu
│   ├── integration/
│   │   └── learner_mixin.py        # điểm gắn duy nhất vào learner của repo gốc
│   ├── runner/
│   │   ├── queue.py                # hàng đợi thí nghiệm, 1 job/GPU
│   │   └── launch.py               # chọn single / ddp / job-parallel
│   └── utils/
│       ├── log.py  seed.py  timing.py  io.py  report.py
│
├── scripts/
│   ├── launch.py                   # lối vào chính: tự quyết định chế độ chạy
│   ├── doctor.py                   # in báo cáo phần cứng + kiểm tra đường dẫn (chạy trên Kaggle)
│   └── collect_results.py          # gom log -> bảng mean ± std
│
├── configs/
│   └── experiments/
│       ├── ablation_core.json      # E0, E4, E5, E7, E8
│       └── ablation_full.json      # E0–E9
│
├── kaggle/
│   ├── bootstrap.py                # clone, symlink dữ liệu, cài phụ thuộc thiếu
│   ├── requirements-kaggle.txt     # KHÔNG chứa torch/torchvision
│   └── notebook_cells.md           # các ô lệnh mẫu để dán vào notebook
│
├── docs/
│   ├── REPO_MAP.md                 # kết quả đọc tĩnh code gốc
│   ├── ARCHITECTURE.md             # sơ đồ module và luồng dữ liệu
│   └── RISK_REGISTER.md            # các điểm dễ lỗi ở lần chạy đầu trên Kaggle
├── .gitignore                      # loại datasets, weights, logs, checkpoint, outputs
└── AGENT_PLAN_v2_KIEN_TRUC_KAGGLE.md
```

---

## 4. Thiết kế từng khối

### 4.1. Nhận diện phần cứng: `hardware/detect.py`

`HardwareProfile` (dataclass, thu thập một lần, in ra log và lưu JSON vào thư mục output):

| Nhóm | Trường | Cách đọc |
|---|---|---|
| Nền tảng | `platform` (kaggle / colab / local), `work_dir`, `disk_free_gb` | biến môi trường `KAGGLE_KERNEL_RUN_TYPE` hoặc tồn tại `/kaggle/working`; `shutil.disk_usage` |
| CPU | `cpu_count` | `len(os.sched_getaffinity(0))` (tôn trọng giới hạn container), fallback `os.cpu_count()` |
| RAM | `ram_gb` | đọc `/proc/meminfo`, fallback `psutil` nếu có; tôn trọng cgroup nếu đọc được |
| Phần mềm | `torch_version`, `cuda_version`, `timm_version`, `nccl_available` | `torch.version.cuda`, `torch.distributed.is_nccl_available()` |
| GPU (danh sách) | `index`, `name`, `total_mem`, `free_mem`, `cc_major/minor`, `bf16_ok`, `tf32_ok`, `p2p_ok` | `torch.cuda.get_device_properties`, `mem_get_info`, `is_bf16_supported`, `can_device_access_peer` |

Quy tắc: mọi truy vấn CUDA bọc `try/except`; nếu không có GPU thì trả hồ sơ CPU và **không crash**.
`homogeneous_gpus` = tất cả GPU cùng tên và cùng dung lượng (quyết định DDP có hợp lý không).

### 4.2. Lập kế hoạch chạy: `hardware/plan.py`

`RuntimePlan` được sinh từ `HardwareProfile` + yêu cầu người dùng (`--mode auto` mặc định; cho phép ép tay).

Trường chính: `mode` (`cpu`/`single`/`ddp`/`jobpar`), `world_size`, `amp_dtype` (`bf16`/`fp16`/`fp32`), `use_grad_scaler`, `per_device_batch`, `grad_accum_steps`, `eval_batch_size`, `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`, `tf32`, `cudnn_benchmark`, `use_compile`, `grad_checkpointing`, `quantum_dtype` (luôn `fp32`).

**Bảng quyết định (mặc định):**

| Hồ sơ | `mode` | AMP | Ghi chú |
|---|---|---|---|
| Không GPU | `cpu` | fp32 | chỉ để `doctor.py`/smoke, cảnh báo rất chậm |
| 1 GPU, compute capability ≥ 8.0 | `single` | bf16, bật TF32 | không cần GradScaler |
| 1 GPU, cc 7.x (T4, V100) | `single` | fp16 + GradScaler | bf16 không được hỗ trợ tốt |
| 1 GPU, cc 6.x (P100) | `single` | fp16 + GradScaler, cho phép lùi fp32 | lợi ích tăng tốc hạn chế |
| ≥ 2 GPU cùng loại, có NCCL | `ddp` hoặc `jobpar` (xem 4.6) | theo cc | |
| ≥ 2 GPU khác loại | `jobpar`, hoặc `single` trên GPU mạnh nhất | theo từng GPU | DDP sẽ chậm theo GPU yếu nhất |
| Không có NCCL hoặc init lỗi | lùi `single` | theo cc | ghi cảnh báo |

**Quy tắc batch (quan trọng):** $\mathcal{L}_{\mathrm{RS}}$ và $\mathcal{L}_{\mathrm{qrel}}$ dựa trên **các cặp trong batch**, nên chia nhỏ batch làm thay đổi tập cặp. Thứ tự ưu tiên:

1. Giữ nguyên **global batch** của cấu hình gốc trên một thiết bị nếu vừa bộ nhớ (xác định bằng OOM-probe).
2. Nếu dùng DDP: gom đặc trưng bằng `all_gather` có gradient để tính loss theo cặp trên batch toàn cục (cờ `--pair_gather`, mặc định **bật** khi `mode=ddp`).
3. Gradient accumulation là phương án cuối, kèm cảnh báo rằng loss theo cặp được tính trên micro-batch.

Global batch lấy từ cấu hình gốc (agent đọc từ `exps/*.json`/`args.sh`); không tự đặt số.
Mặc định **không** tự nhân learning rate theo số GPU (giữ ngữ nghĩa gốc); có cờ `--lr_scale_by_world` tắt sẵn.

### 4.3. OOM-probe: `hardware/probe.py`

Chạy lúc khởi động trên GPU (chỉ khi `--probe` bật, mặc định bật trên Kaggle): dựng batch dữ liệu giả, thử forward+backward với batch tăng dần, bắt `torch.cuda.OutOfMemoryError`, lùi lại mức an toàn (trừ biên khoảng 10–15%). Kết quả ghi vào `RuntimePlan.per_device_batch`. Dọn `torch.cuda.empty_cache()` sau probe. Nếu probe tự lỗi thì dùng mặc định bảo thủ và ghi log.

### 4.4. Ngữ cảnh chạy: `runtime/`

- `RuntimeContext`: `device`, `rank`, `local_rank`, `world_size`, `is_main`, `autocast()` (context manager), `scaler`, `dtype`.
- `precision.py`: bọc autocast; **khối lượng tử luôn chạy ngoài autocast ở fp32** (fidelity nhạy với sai số).
- `optimize.py` (mỗi mục có fallback và cờ riêng, mặc định chọn phương án an toàn):
  - `torch.backends.cuda.matmul.allow_tf32` và `cudnn.allow_tf32` khi cc ≥ 8.0.
  - `cudnn.benchmark = True` (kích thước ảnh cố định 224).
  - Attention hợp nhất (`F.scaled_dot_product_attention`) nếu timm/PyTorch hỗ trợ: chỉ ghi nhận, không tự vá.
  - `torch.compile`: **mặc định tắt**; bật bằng cờ, bọc `try/except`, lùi eager nếu lỗi.
  - Ép kiểu backbone đóng băng sang dtype AMP (`--freeze_cast`): mặc định tắt, tiết kiệm bộ nhớ khi bật.
  - Gradient checkpointing: chỉ bật khi probe cho thấy không đủ bộ nhớ ở batch gốc.
  - Backbone đóng băng và nhánh $f^{t-1}$ chạy trong `torch.inference_mode()`.

### 4.5. Dữ liệu và trọng số: `data/`

- `registry.py`: ánh xạ tên dataset → danh sách đường dẫn ứng viên, theo thứ tự: `$DATA_ROOT/<tên>`, `./datasets/<tên>`, và (trên Kaggle) quét `/kaggle/input/*` theo mẫu tên thư mục. Khi tìm thấy ở `/kaggle/input`, **tạo symlink vào `./datasets/<tên>`** để mã gốc (đọc `datasets/` tương đối) chạy không cần sửa. Thiếu dataset: dừng với thông báo nêu rõ tên Kaggle Dataset cần gắn.
- `weights.py`: nếu mã gốc tải ViT-B/16-IN21K qua timm/HF (cần internet), cung cấp đường tải offline: tìm file trọng số trong `/kaggle/input/*`, đặt vào thư mục cache (`TORCH_HOME`/`HF_HOME`) trước khi dựng mô hình. Cách mã gốc nạp trọng số phải được ghi trong `REPO_MAP.md`.
- `loaders.py`:
  - `num_workers = min(cpu_count // world_size, trần)`, `persistent_workers=True` nếu workers > 0, `prefetch_factor` 2–4, `pin_memory` khi có GPU.
  - CPU thường là nút thắt trên Kaggle (ít nhân): khi giải mã/augmentation chậm, workers là điểm chỉnh đầu tiên.
  - (Tùy chọn, giai đoạn cuối, mặc định tắt) `--data_backend uint8_cache`: tiền xử lý ảnh về uint8 224×224, lưu RAM/đĩa và augmentation trên GPU. **Đổi ngữ nghĩa augmentation so với gốc**, phải ghi rõ trong log và tài liệu.

### 4.6. Đa GPU: `distributed/`, `training/step_module.py`, `runner/`

Hai chiến lược, `--mode auto` chọn theo số thí nghiệm trong hàng đợi:

**(i) Job-parallel (ưu tiên khi chạy ablation):** mỗi GPU chạy **một thí nghiệm độc lập** (`CUDA_VISIBLE_DEVICES=i`). Không giao tiếp giữa tiến trình, hiệu suất gần tuyến tính, và không làm đổi ngữ nghĩa loss theo cặp. Dùng khi số job ≥ số GPU và mỗi job vừa một GPU.

**(ii) DDP cho một thí nghiệm:** dùng khi chỉ có một thí nghiệm lớn cần chạy nhanh hơn. Khởi chạy bằng `torchrun --standalone --nproc_per_node=N` từ `launch.py` qua `subprocess` (hoạt động trong notebook).

**Hợp đồng kiến trúc cho DDP (bắt buộc tuân thủ, vì không có test để bắt lỗi):**

1. **Một mô-đun bọc duy nhất.** `StepModule.forward(batch) -> dict các loss` chứa: backbone + adapter, classifier, aligner, tham số lượng tử. Chỉ **StepModule** được bọc `DistributedDataParallel`. Mô-đun huấn luyện nào gọi **ngoài** forward của DDP sẽ **không được đồng bộ gradient**. Nhánh $f^{t-1}$ (đóng băng, no-grad) được phép nằm ngoài.
2. **Không tham số thừa.** Chỉ khởi tạo aligner/mạch lượng tử khi cờ bật; mọi tham số khả huấn luyện phải tham gia loss ở mỗi bước, để dùng `find_unused_parameters=False`. Hinge cho giá trị 0 vẫn phải nằm trong đồ thị.
3. **Bọc lại DDP ở đầu mỗi task** (tập tham số khả huấn luyện đổi theo task). `broadcast_buffers=False` (ViT dùng LayerNorm, không có BatchNorm, nên không cần SyncBN).
4. Khối lượng truyền thông nhỏ (adapter khoảng $2\cdot 768\cdot 64\cdot 12\approx 1.2$ triệu tham số), nên mức tăng tốc bị giới hạn bởi tính toán chứ không phải băng thông.
5. **Thống kê lớp phải gộp chính xác.** `stats/accumulators.py` cộng dồn theo từng rank: số mẫu, tổng đặc trưng, tổng $\sum xx^\top$ theo lớp (dùng float64), rồi `all_reduce(SUM)`, sau đó mới tính prototype và covariance. **Không** lấy trung bình của các trung bình. Ước lượng drift làm tương tự.
6. **Retrain classifier trên feature Gaussian:** chạy ở rank 0 rồi `broadcast` trọng số (tránh phân kỳ do không tất định giữa các rank). Các rank còn lại chờ ở `barrier` với timeout dài.
7. **Đánh giá:** dùng sampler đánh giá **không padding** (hoặc gom kèm chỉ số mẫu rồi loại trùng) để số mẫu đúng; gom số đúng/tổng bằng `all_reduce`; chỉ rank 0 ghi log.
8. **Seed:** khởi tạo mô hình giống nhau mọi rank (DDP broadcast lúc bọc); seed augmentation = `seed + rank`; gọi `sampler.set_epoch(e)` mỗi epoch.
9. **Khởi tạo process group** (`distributed/setup.py`): `torch.cuda.set_device(local_rank)`, timeout dài (ví dụ 30 phút để chịu được bước tính thống kê), nếu `init_process_group` lỗi thì thử lại với `NCCL_P2P_DISABLE=1`, nếu vẫn lỗi thì lùi `single` và ghi cảnh báo. Luôn `destroy_process_group()` ở khối `finally`.
10. Loss theo cặp dưới DDP: dùng `all_gather` có gradient (`torch.distributed.nn.functional.all_gather`) khi `--pair_gather` bật; nếu không bật thì tính cục bộ và log cảnh báo về thay đổi ngữ nghĩa.

**Hàng đợi (`runner/queue.py`):** đọc `configs/experiments/*.json` (mỗi mục: `id`, ghi đè cờ, danh sách seed); bộ lập lịch giữ N worker (mỗi GPU một worker); thư mục `out/<id>_s<seed>/` có file trạng thái `DONE`/`FAILED`; **bỏ qua job đã `DONE`** (cho phép chạy lại sau khi hết phiên); `--time_budget_h` ngừng phát job mới khi gần hết thời gian phiên; log lỗi giữ nguyên, không nuốt exception.

### 4.7. Tích hợp vào learner gốc: `integration/learner_mixin.py`

- **Một điểm gắn duy nhất.** Ưu tiên mixin/subclass của learner RSIAT, đăng ký qua cơ chế factory sẵn có của repo (ghi vị trí ở `REPO_MAP.md`). Chỉ khi buộc phải mới sửa trực tiếp file gốc, và mỗi lần sửa phải ghi vào danh sách "Các điểm đã chạm" trong `ARCHITECTURE.md`.
- Việc cần làm trong mixin: dựng `RuntimeContext`; thay mọi `.cuda()`/`device` cứng bằng `ctx.device`; thay `DataParallel` (nếu mã gốc/PILOT dùng) bằng cơ chế của `RuntimeContext`; dựng `StepModule`; gọi các loss mới theo cờ; gọi checkpoint cuối mỗi task.
- Việc refactor mã gốc cho phép chạy trên mọi cấu hình thiết bị là **bắt buộc**, vì PILOT thường giữ danh sách thiết bị trong `args`. Agent phải `grep` các mẫu: `.cuda(`, `torch.device(`, `DataParallel`, `device_ids`, `CUDA_VISIBLE_DEVICES`, đường dẫn tuyệt đối, và xử lý hết.

### 4.8. Checkpoint và tiếp tục chạy: `training/checkpoint.py`

Kaggle giới hạn thời gian phiên và dung lượng đĩa ghi, nên **phải** có resume.

- Lưu cuối **mỗi task**: trọng số khả huấn luyện (adapter, classifier, aligner), prototype, covariance, trạng thái bù drift, thứ tự lớp, lịch sử metric, trạng thái RNG, băm (hash) cấu hình. Chỉ lưu phần khả huấn luyện, không lưu backbone.
- Ghi nguyên tử (ghi file tạm rồi đổi tên); chỉ rank 0 ghi; giữ tối đa 2 checkpoint gần nhất; `latest.json` trỏ tới checkpoint mới nhất.
- `--resume auto`: tìm trong thư mục output, và trong `--resume_from <đường dẫn>` (để trỏ tới output của phiên Kaggle trước được gắn làm Dataset). Từ chối resume nếu băm cấu hình không khớp, trừ khi có `--force_resume`.

### 4.9. Khối lượng tử: `quantum/`

Thiết kế như tài liệu QR-RSIAT: mạch nông ($q\in\{4,8,12\}$, $l_q\in\{1,2,3\}$, mặc định $q=8$, $l_q=2$), mô phỏng tensor thực, mã hóa bằng lớp tuyến tính $768\to q\cdot l_q$ góc, centering bật/tắt bằng cờ, mọi hàm hỗ trợ autograd, không vòng lặp Python theo mẫu. Bốn kernel (`quantum`, `cosine`, `rbf`, `mlp`) **cùng giao diện** `kernel(Z) -> [B,B]` để ablation công bằng; chỉnh số tham số của `mlp` cho tương đương.
Aligner: $P^{t}(f)=f+W_{\mathrm{up}}\,\mathrm{Readout}(\mathrm{PQC}(W_{\mathrm{down}}f))$, $W_{\mathrm{up}}^{(0)}=0$ đặt lại ở đầu mỗi task $t>1$.
Dưới DDP: kernel $B\times B$ tính trên batch toàn cục sau `all_gather` (mục 4.6, điểm 10); bộ nhớ trạng thái $B\times 2^{q}$ rất nhỏ.

### 4.10. Cấu hình và báo cáo: `config/`, `utils/report.py`

- `schema.py` kiểm tra hợp lệ ngay khi nạp (`q`, `l_q`, `k`, `tau`, tổ hợp cờ vô lý như `--kernel` khác `quantum` mà `--aligner qhybrid` thiếu tham số...), thông báo lỗi nêu rõ cờ nào sai và giá trị hợp lệ.
- Mỗi lần chạy ghi vào thư mục output: `hardware.json`, `plan.json`, `config.json`, log từng task, `metrics.json`, thời gian mỗi giai đoạn, số tham số khả huấn luyện (tách riêng phần lượng tử).
- Toàn bộ `main` bọc trong `try/except`: khi lỗi, ghi traceback + hồ sơ phần cứng + kế hoạch chạy vào `out/<id>/FAILED.txt` rồi mới ném lại exception.

---

## 5. Lộ trình viết mã (không có bước test)

| Giai đoạn | Việc | Sản phẩm |
|---|---|---|
| G1 | Đọc tĩnh repo gốc | `docs/REPO_MAP.md` (đường dẫn:dòng cho các điểm ở mục 1, danh sách mọi chỗ dùng thiết bị cứng, cách nạp dữ liệu/trọng số, cách tính prototype/covariance, vòng eval, "Vấn đề mở") |
| G2 | Nền tảng | `qrsiat/config`, `hardware`, `runtime`, `utils` |
| G3 | Phân tán và dữ liệu | `distributed`, `data`, `stats` |
| G4 | Khối lượng tử và loss | `quantum/` đầy đủ |
| G5 | Vòng huấn luyện | `training/` (`StepModule`, `loop`, `schedule`, `checkpoint`, `eval`) |
| G6 | Tích hợp | `integration/learner_mixin.py` + các hook tối thiểu vào file gốc, refactor thiết bị |
| G7 | Chạy và đóng gói | `runner/`, `scripts/`, `kaggle/`, `configs/experiments/`, `.gitignore` |
| G8 | Tài liệu và rà soát tĩnh | `ARCHITECTURE.md`, `RISK_REGISTER.md`, checklist mục 6 |

Mỗi giai đoạn kết thúc bằng một commit và một tóm tắt ngắn: đã viết gì, giả định nào chưa xác minh.

**Công cụ chẩn đoán chạy trên Kaggle (không phải test, là tính năng thời gian chạy):**
- `scripts/doctor.py`: in hồ sơ phần cứng, `RuntimePlan` dự kiến, kiểm tra đường dẫn dataset/trọng số, kiểm tra NCCL, in cảnh báo; **không** huấn luyện.
- Cờ `--smoke`: 2 task, vài bước mỗi task, để người dùng kiểm tra nhanh trên Kaggle trước khi chạy dài.

---

## 6. Checklist rà soát tĩnh trước khi đẩy lên (agent tự đọc lại mã, không cần chạy)

- [ ] `grep` không còn `.cuda(`, `torch.device("cuda`, `DataParallel`, `device_ids`, đường dẫn tuyệt đối, `/home/` trong mã (trừ `platform_kaggle.py`).
- [ ] Mọi import trong mỗi file trỏ tới module thật tồn tại (kiểm tra bằng `grep`/đọc cây thư mục); không có import vòng giữa `qrsiat/*`.
- [ ] Chữ ký hàm khớp giữa nơi gọi và nơi định nghĩa (đặc biệt `RuntimeContext`, `StepModule`, kernel).
- [ ] Mô-đun huấn luyện nào cũng nằm **trong** `StepModule.forward` (hợp đồng DDP, mục 4.6 điểm 1).
- [ ] Aligner/mạch lượng tử chỉ được tạo khi cờ bật (tránh tham số thừa).
- [ ] Khối lượng tử chạy ngoài autocast, fp32.
- [ ] Cờ mặc định = hành vi RSIAT gốc.
- [ ] Đường suy luận không gọi aligner hay mạch lượng tử.
- [ ] Mọi chỗ ghi file chỉ do rank 0 thực hiện; mọi `barrier`/collective được mọi rank cùng gọi (không nằm trong nhánh `if is_main`).
- [ ] Mỗi tối ưu rủi ro (compile, freeze_cast, uint8_cache) có `try/except` và đường lùi.
- [ ] `requirements-kaggle.txt` không chứa `torch`, `torchvision`.
- [ ] `.gitignore` loại `datasets/`, `*.pth`, `*.pt`, `logs/`, `outputs/`, `__pycache__/`.
- [ ] `RISK_REGISTER.md` liệt kê các điểm dễ lỗi nhất ở lần chạy đầu (nạp trọng số offline, symlink dataset, NCCL, loss theo cặp dưới DDP, resume).

---

## 7. Quy trình đẩy lên GitHub và chạy trên Kaggle

### 7.1. Đẩy mã (làm ở máy hiện tại, sau khi G8 xong)

```bash
git remote add mine https://github.com/<tài-khoản>/<tên-repo>.git
git push mine qr-rsiat
```

Repo riêng tư cần token. Không commit token vào mã.

### 7.2. Chuẩn bị trên Kaggle

1. Tạo Notebook mới, chọn **Accelerator GPU** (chọn loại có 2 GPU nếu muốn thử đa GPU), bật **Internet** để `git clone` (Kaggle có thể yêu cầu xác minh tài khoản; kiểm tra điều kiện hiện hành).
2. Đưa dữ liệu lên dạng **Kaggle Dataset** và **Add Data** vào notebook: CIFAR-100, CUB-200, ImageNet-R, ImageNet-A, VTAB, OmniBenchmark (tùy thí nghiệm cần chạy). Nếu mã gốc cần trọng số ViT-B/16-IN21K ngoại tuyến thì đưa lên thành một Dataset riêng.
3. Repo riêng tư: lưu token GitHub vào **Kaggle Secrets**.
4. Kiểm tra giới hạn hiện hành của Kaggle (thời gian phiên, dung lượng `/kaggle/working`, số nhân CPU, loại GPU) vì các giá trị này có thể thay đổi; `doctor.py` sẽ in ra những gì phiên đang có.

### 7.3. Các ô lệnh mẫu (`kaggle/notebook_cells.md`)

```python
# Ô 1: lấy mã (repo công khai)
!git clone --depth 1 -b qr-rsiat https://github.com/<tài-khoản>/<tên-repo>.git /kaggle/working/repo
%cd /kaggle/working/repo

# Với repo riêng tư, dùng Kaggle Secrets:
# from kaggle_secrets import UserSecretsClient
# tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
# !git clone --depth 1 -b qr-rsiat https://{tok}@github.com/<tài-khoản>/<tên-repo>.git /kaggle/working/repo
```

```python
# Ô 2: phụ thuộc thiếu (không cài lại torch)
!pip install -q -r kaggle/requirements-kaggle.txt
```

```python
# Ô 3: chẩn đoán phần cứng và đường dẫn, chưa huấn luyện
!python scripts/doctor.py --data_root /kaggle/input
```

```python
# Ô 4: chạy thử nhanh
!python scripts/launch.py --smoke --config configs/experiments/ablation_core.json \
    --data_root /kaggle/input --out /kaggle/working/outputs
```

```python
# Ô 5: chạy thật, tự chọn single / ddp / job-parallel theo số GPU
!python scripts/launch.py --mode auto --probe \
    --queue configs/experiments/ablation_core.json \
    --data_root /kaggle/input --out /kaggle/working/outputs \
    --time_budget_h 10 --resume auto
```

```python
# Ô 6: gom kết quả
!python scripts/collect_results.py --out /kaggle/working/outputs
```

### 7.4. Phiên dài và tiếp tục chạy

- Phiên bị cắt thì mất mọi thứ chưa lưu. Lưu **Output** của phiên (lưu phiên hoặc commit notebook), rồi ở phiên sau gắn Output đó làm Dataset và chạy với `--resume_from /kaggle/input/<tên-output-trước>`.
- Hàng đợi bỏ qua job đã `DONE`, nên chạy lại cùng lệnh sẽ tiếp tục phần còn thiếu.
- Chỉ giữ trong `/kaggle/working` những gì cần: log, metrics, checkpoint 2 task gần nhất.

---

## 9. Điều agent không được làm

- Không cài hoặc yêu cầu cài lại `torch`/`torchvision`.
- Không hardcode GPU, đường dẫn tuyệt đối, hay số tiến trình.
- Không đổi giao thức đánh giá, thứ tự lớp, chia train/test.
- Không bật mặc định các tối ưu có rủi ro chưa kiểm chứng (`torch.compile`, `freeze_cast`, `uint8_cache`).
- Không đưa token, khóa API hay dữ liệu lên GitHub.
- Không khẳng định "đã chạy được" hay đưa số liệu hiệu năng: mã chưa từng được thực thi.
