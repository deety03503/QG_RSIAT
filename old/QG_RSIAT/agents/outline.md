# AGENT_PLAN: Đọc hiểu và chỉnh sửa repo RSIAT để triển khai QR-RSIAT

> Tài liệu này dành cho coding agent (Claude Code hoặc tương đương). Đọc toàn bộ trước khi chạy bất kỳ lệnh nào.
> Repo gốc: https://github.com/zjrzjrz/RSIAT (MIT license, xây trên PILOT và SSIAT).

---

## 0. Sứ mệnh và ràng buộc

**Mục tiêu:** thêm phương pháp QR-RSIAT vào repo RSIAT dưới dạng các **cờ cấu hình có thể bật/tắt**, sao cho khi tắt hết cờ thì hành vi và kết quả **giống hệt RSIAT gốc**.

**QR-RSIAT gồm bốn thay đổi về phương pháp so với RSIAT:**

| Mã | Thay đổi | Cờ đề xuất |
|---|---|---|
| A | Thay RAE Projector bằng bộ căn chỉnh lai cổ điển–lượng tử, khởi tạo đồng nhất | `--aligner {rae,qhybrid}` |
| B | Thêm $\mathcal{L}_{\mathrm{qrel}}$: khớp ma trận kernel fidelity giữa $f^{t-1}$ và $f^{t}$ | `--lambda_qrel`, `--kernel {quantum,cosine,rbf,mlp}` |
| C | Thay $\mathcal{L}_{\mathrm{orth}}$ bằng $\mathcal{L}_{\mathrm{orth}}^{q}$ (top-$k$ + trọng số $\alpha$ + hinge) | `--orth {plain,qweighted}` |
| D | Tùy chọn: dùng kernel lượng tử trong $\mathcal{L}_{\mathrm{RS}}$ ở base task | `--rs_kernel {cosine,quantum}` |

**Hạ tầng thực nghiệm** (không phải thay đổi phương pháp A–D):

| Hạng mục | Thay đổi | Cấu hình |
|---|---|---|
| E | Chạy một lần huấn luyện trên nhiều GPU; cấu hình số worker nạp dữ liệu | `device: ["0", "1", ...]`, `--num_worker` |
| F | Điều khiển seed chạy qua config/CLI để phân tán thí nghiệm qua nhiều phiên | `seed` |

**Ràng buộc bắt buộc:**

1. Không đổi giao thức đánh giá, thứ tự lớp (seed 1993 cố định), chia train/test, hay cách tính $\bar{\mathcal{A}}$ và $\mathcal{A}_B$. `seed` cấu hình chỉ điều khiển khởi tạo mô hình, sampler/DataLoader và các nguồn ngẫu nhiên khác; không dùng nó để đổi thứ tự lớp.
2. Bộ căn chỉnh và mạch lượng tử **chỉ dùng khi train**. Đường suy luận (PTM đóng băng + adapter chung + classifier) phải không đổi. Phải có test khẳng định điều này.
3. Mỗi bước nhỏ một commit riêng, trên nhánh `qr-rsiat`. Không sửa trực tiếp `main`.
4. Không cài thêm thư viện lượng tử (PennyLane, Qiskit). Tự viết statevector simulator bằng PyTorch (mục 5).
5. Không tự bịa số liệu. Chỉ báo cáo số do chính các lần chạy tạo ra.
6. Nếu một giả định trong tài liệu này sai so với code thật, **ghi lại vào `REPO_MAP.md` và điều chỉnh kế hoạch**, đừng cố ép code theo tài liệu.
7. Multi-GPU phải được kiểm chứng là có chia batch và đồng bộ gradient thực sự; chỉ bọc `nn.DataParallel` nhưng vẫn gọi module gốc không được tính là đã hỗ trợ.
8. `num_worker` chỉ điều khiển số worker CPU của mỗi `DataLoader`, không phải số GPU. Giá trị `0` phải hợp lệ để debug hoặc chạy trên môi trường hạn chế worker.
9. Máy cá nhân không có GPU: dùng máy này cho đọc hiểu code, cấu hình, unit test và kiểm tra CPU; dùng Kaggle cho smoke test GPU, baseline, benchmark và kiểm chứng multi-GPU.
10. Môi trường thực nghiệm dự kiến là Kaggle với 30 GB RAM hệ thống và 2 GPU T4. Đây là giới hạn mục tiêu, cần ghi nhận tài nguyên thực tế mỗi phiên. RAM hệ thống không phải VRAM; VRAM của từng T4 là riêng, không cộng gộp. Không khẳng định có hai T4 cho tới khi kiểm tra runtime.
11. Dataset ImageNet-R, ImageNet-A, CUB, VTAB và weights ViT-B/16-IN21K được người dùng xác nhận có sẵn trong Kaggle Input. Không tự tải lại nếu input đã gắn; kiểm tra đường dẫn và khả năng đọc trước khi chạy.
12. Phiên Kaggle có thể kết thúc giữa thí nghiệm. Lưu checkpoint theo task, config đã merge, trạng thái seed, log và kết quả thô dưới `/kaggle/working`; sau mỗi phiên, Save Version/commit notebook và tải Output để lưu bền trước khi kết thúc phiên.

---

## 1. Những gì đã biết về repo (và những gì chưa)

**Đã xác minh từ repo cục bộ (ghi lại đường dẫn và dòng chính xác trong `REPO_MAP.md`):**

- Cây thư mục gốc: `data/`, `exps/`, `images/`, `logs/adapter/`, `models/`, `network/`, `utils/`, `LICENSE`, `README.md`, `args.sh`, `main.py`, `requirements.txt`, `trainer.py`.
- README khai báo Ubuntu 20.04, Python 3.10, CUDA 11.8; tuy nhiên `requirements.txt` ghim PyTorch `2.8.0+cu126` và torchvision `0.23.0+cu126`. Đây là điểm không nhất quán cần xác minh với môi trường thực tế trước khi baseline, không nên coi cả hai mô tả là đã tương thích.
- Chạy qua `args.sh` (mỗi dataset một lệnh), log lưu vào `./logs`.
- Dữ liệu đặt ở `datasets/` (cifar-100-python, cub, imagenet-a, imagenet-r, omnibenchmark, vtab).
- Repo kế thừa từ **PILOT** (LAMDA-PILOT) và **SSIAT**.
- `main.py` hiện chỉ nhận `--config`; cấu hình JSON đặt `device` dưới dạng danh sách chuỗi, ví dụ `["0"]`.
- `models/RSIAT_adapter.py` đã có nhánh tạo `nn.DataParallel` cho nhiều GPU, nhưng `_network_module_ptr` và các lệnh gọi trực tiếp `.extract_vector()`/`.fc()` có thể bỏ qua wrapper; các truy cập `self._network.convnet` sau khi bọc cũng cần kiểm tra vì thuộc tính custom nằm trên `.module`. Cần kiểm tra đường forward và optimizer cụ thể, không xem việc tạo wrapper là bằng chứng đủ cho multi-GPU.
- Learner tạo train/test `DataLoader` với `num_workers=8`; `models/base.py` có loader khác dùng `num_workers=4`. Biến `num_workers = 8` trong learner hiện không được các loader dùng.
- `trainer.py` chuyển `device` từ cấu hình thành `torch.device`, learner lấy GPU đầu tiên làm device chính và dùng danh sách GPU cho `DataParallel`. Cần kiểm tra kiểu dữ liệu và xử lý CPU/GPU ở cả hai nơi.
- `main.merge_configs()` hiện để JSON ghi đè tham số CLI; cần sửa quy tắc gộp để `--num_worker` truyền từ CLI có thể override cấu hình JSON.

**Chưa xác minh (agent phải tự khám phá, mục 3):**

- Tên file/class cụ thể trong `models/`, `network/`, `utils/`.
- Cấu trúc file JSON trong `exps/`.
- Vị trí chính xác của: cosine loss, L_RS, warm-up, RAE projector, L_align, L_orth, thu thập prototype/covariance, bù drift, retrain classifier.

**Giả định dựa trên cấu trúc PILOT** (cần kiểm tra, không được coi là chắc chắn):

- `models/` chứa các lớp learner, mỗi lớp có `incremental_train`, `_train`, `eval_task`, `after_task`.
- `network/` chứa backbone ViT có adapter và lớp mạng gói backbone + classifier.
- `utils/` chứa `data_manager`, `toolkit`, `factory`.
- `exps/*.json` chứa hyperparameter theo dataset.

---

## 2. Giai đoạn 0: Chuẩn bị và kiểm tra nền

**Việc cần làm:**

1. `git clone` repo, tạo nhánh `qr-rsiat`, ghi lại commit hash gốc vào `REPO_MAP.md`.
2. Đọc `README.md`, `args.sh`, `requirements.txt`. Liệt kê các tham số dòng lệnh.
3. Trên Kaggle, trước khi cài hoặc nâng cấp package, ghi nhận Python, PyTorch, CUDA runtime, driver, GPU model/count và VRAM của từng GPU; kiểm tra tương thích với requirements. README ghi Python 3.10/CUDA 11.8 nhưng requirements ghim PyTorch `2.8.0+cu126` và torchvision `0.23.0+cu126`, nên không cài requirements máy móc. Chỉ thay package khi có lỗi tương thích xác định và ghi lại phiên bản thực tế.
4. Trên máy cá nhân không GPU, hoàn thành các bước đọc code, kiểm tra cấu hình và unit test CPU khi có thể; không yêu cầu baseline GPU chạy tại máy này.
5. Trên Kaggle, kiểm tra input datasets/backbone đã xác nhận, đường dẫn, quyền đọc và dung lượng khả dụng. Nếu input thực tế bị thiếu hoặc không đọc được, dừng và hỏi người dùng thay vì tự tải tùy ý.
6. Chạy **smoke test** nhỏ: dataset phù hợp, ít epoch, 2 task; sau đó xác nhận cấu hình một GPU và cấu hình hai GPU nếu runtime nhận đủ hai T4.
7. Ghi riêng RAM hệ thống và VRAM từng GPU; xác minh hai GPU thực sự hiện diện trong runtime trước multi-GPU. Nếu runtime chỉ có một GPU, ghi rõ kiểm chứng multi-GPU chưa thực hiện, không báo là đã đạt.

**Tiêu chí hoàn thành:** smoke test chạy hết trên Kaggle, có log và thông tin runtime, không lỗi. Lưu log vào `logs/baseline_smoke/` và đồng bộ bản sao dưới `/kaggle/working` để tải Output.

---

## 3. Giai đoạn 1: Đọc hiểu và lập bản đồ code

**Đầu ra:** file `REPO_MAP.md` trả lời đầy đủ các câu hỏi dưới đây, mỗi câu kèm `đường dẫn:dòng`.

### 3.1. Luồng thực thi

- `main.py` đọc tham số thế nào, gọi `trainer.py` ra sao?
- `trainer.py` vòng lặp theo task như thế nào? Khi nào gọi `after_task`, `eval`?
- Learner nào được dùng cho RSIAT (xem `exps/` và `utils/factory`)?

### 3.2. Các thành phần cần chạm vào

Với mỗi mục, ghi: file, hàm, dòng, và các tensor đầu vào/ra.

| # | Cần tìm | Mục đích |
|---|---|---|
| 1 | Cosine loss (margin $m$, scale $s$) | Hiểu logits và nhãn |
| 2 | $\mathcal{L}_{\mathrm{RS}}$ và lịch warm-up $\lambda_{\mathrm{RS}}(e)$ | Nơi gắn tùy chọn D |
| 3 | RAE Projector (định nghĩa module, khởi tạo W_up = 0, kích thước 64 và $d_p$) | Nơi thay bằng aligner lai |
| 4 | Cách lấy $f^{t-1}$: có mô hình đóng băng riêng hay copy adapter $A^{t-1}$ không? | Cần cho $\mathcal{L}_{\mathrm{qrel}}$ |
| 5 | $\mathcal{L}_{\mathrm{align}}$ | Giữ làm thành phần phụ |
| 6 | $\mathcal{L}_{\mathrm{orth}}$ (prototype cũ, ánh xạ qua projector, ℓ2-norm) | Nơi thay bằng $\mathcal{L}_{\mathrm{orth}}^{q}$ |
| 7 | Lưu prototype và covariance sau mỗi task | Nguồn $\hat p_c$ |
| 8 | Bù drift (semantic shift) cho prototype cũ | Có dùng cho $\mathcal{L}_{\mathrm{orth}}$ không? |
| 9 | Retrain classifier bằng feature Gaussian | Giữ nguyên |
| 10 | Đường suy luận (`eval`, `_eval_cnn`, ...) | Phải không bị ảnh hưởng |
| 11 | Optimizer, scheduler, tham số nào được huấn luyện ở task $t>1$ | Quyết định thêm tham số mạch vào đâu |
| 12 | Cách đặt seed, cách log metric | Tái lập và so sánh |
| 13 | Forward qua `DataParallel`: logits, features, loss và trích xuất feature | Xác nhận mỗi đường cần tính toán song song đều gọi wrapper |
| 14 | Mọi vị trí tạo `DataLoader` | Đưa `num_worker` vào train/test/feature extraction/class-mean/classifier calibration |
| 15 | Thiết lập GPU/device và batch size | Chọn GPU chính, kiểm tra GPU khả dụng và điều kiện batch nhỏ |

### 3.3. Lệnh khám phá gợi ý

```bash
git ls-files | head -200
grep -rn "class .*Learner\|class .*Net\|class .*Adapter" --include=*.py .
grep -rn -i "orth\|align\|projector\|autoencoder\|rs_loss\|warm" --include=*.py .
grep -rn -i "prototype\|proto\|cov\|drift" --include=*.py .
grep -rn "def incremental_train\|def _train\|def after_task\|def eval" --include=*.py .
cat args.sh; ls exps; head -50 exps/*.json
```

### 3.4. Câu hỏi mở phải trả lời trước khi sang giai đoạn sau

- Feature đưa vào projector có đã ℓ2-norm không, chiều bao nhiêu ($d=768$)?
- $\mathcal{L}_{\mathrm{orth}}$ gốc dùng prototype đã bù drift hay chưa bù?
- Projector bị bỏ khi suy luận bằng cách nào (không gọi, hay xóa khỏi state)?
- Có chỗ nào trong code lệch so với công thức trong paper không (ghi lại, không tự sửa)?
- `DataParallel` hiện có tham gia vào forward có gradient hay bị bypass qua module gốc?
- Có bao nhiêu `DataLoader` trong pipeline, và `num_worker=0` có hoạt động ở tất cả loader không?
- Tham số CLI có override JSON không? Nếu không, cần sửa quy tắc merge để chỉ giá trị CLI được truyền rõ ràng mới override JSON.

---

## 4. Giai đoạn 2: Tái lập baseline

1. Chạy RSIAT gốc trên **IN-R B0I20** và **IN-A B0I20** với cấu hình trong `args.sh`.
2. So với bảng 1 của paper: IN-R $\bar{\mathcal{A}}/\mathcal{A}_B$ khoảng 86.92/82.75, IN-A khoảng 74.89/66.23.
3. Dùng seed cấu hình cho các nguồn ngẫu nhiên, nhưng giữ thứ tự lớp theo seed 1993. Chạy tối thiểu 3 seed cho baseline chính, báo cáo mean ± std. Chấp nhận lệch vài phần mười điểm do phần cứng/seed; nếu lệch lớn so với paper thì **báo cáo và dừng** để điều tra, không đi tiếp trên baseline chưa tái lập.
4. Các lượt baseline IN-R B0I20 và IN-A B0I20 được dùng làm E0 tương ứng trong ma trận ở mục 7, tránh chạy trùng. Lưu kết quả vào `results/baseline.md` và `results/raw/`.

---

## 5. Giai đoạn 3: Module lượng tử độc lập (chưa gắn vào pipeline)

Tạo thư mục mới `quantum/`, **không sửa file gốc** ở giai đoạn này.

```
quantum/
├── __init__.py
├── simulator.py      # statevector: RY, CNOT chain, đo kỳ vọng Pauli
├── feature_map.py    # d=768 -> góc q qubit (chiều, chuẩn hóa, centering)
├── kernels.py        # fidelity / cosine / rbf / mlp kernel cùng giao diện
├── aligner.py        # QHybridAligner: skip + W_up(=0) * Readout(PQC(W_down f))
├── losses.py         # qrel_loss, qorth_loss
└── tests/
```

### 5.1. Quyết định thiết kế

- **Chỉ dùng cổng RY và CNOT** nên biên độ luôn là số thực. Khi đó có thể mô phỏng bằng tensor thực `[B, 2**q]`, fidelity giữa hai trạng thái là $(\langle a, b\rangle)^2$. Rất rẻ với $q\le 12$.
- Mã hóa: một lớp tuyến tính (hoặc PCA cố định) ánh xạ $768 \to q\cdot l_q$ góc. Tham số này thuộc mô-đun lượng tử, ghi rõ trong log để tính số tham số.
- Centering: trừ trung bình batch (hoặc trung bình prototype) trước khi mã hóa. Cho phép bật/tắt bằng cờ.
- Mạch nông: $l_q\in\{1,2,3\}$, $q\in\{4,8,12\}$, mặc định $q=8$, $l_q=2$.
- Mọi hàm nhận `torch.Tensor`, hỗ trợ autograd, chạy được trên GPU, không có vòng lặp Python theo mẫu.

### 5.2. Test bắt buộc (`quantum/tests/`)

| Test | Nội dung | Điều kiện đạt |
|---|---|---|
| T1 | Chuẩn hóa trạng thái | $\lVert\psi\rVert_2 = 1$ sai số $<10^{-6}$ |
| T2 | Fidelity nằm trong $[0,1]$, đối xứng, $F(\psi,\psi)=1$ | Đạt |
| T3 | So khớp với mô phỏng tham chiếu bằng NumPy cho $q\le 4$ | Sai số $<10^{-5}$ |
| T4 | **Khởi tạo đồng nhất:** $P^{t}(f)=f$ khi $W_{\mathrm{up}}=0$ | Chính xác tuyệt đối |
| T5 | Kiểm tra gradient bằng `torch.autograd.gradcheck` (double) | Đạt |
| T6 | Gradient norm theo $q$ và $l_q$ (phát hiện concentration) | Ghi lại bảng, cảnh báo nếu $<10^{-8}$ |
| T7 | Thời gian forward với $B=32$, $q=12$ | Ghi lại số đo trên CPU/GPU cụ thể; mục tiêu dưới vài ms chỉ là tham khảo, không phải tiêu chí pass cứng |

**Tiêu chí hoàn thành:** toàn bộ T1–T5 pass. Commit riêng.

---

## 6. Giai đoạn 4: Gắn vào pipeline (từng cờ một)

Nguyên tắc: **mỗi cờ một commit, mỗi commit có một lần chạy kiểm tra hồi quy** (tắt cờ phải cho kết quả như baseline).

### Bước 4.1: Hạ tầng cấu hình

- Thêm các cờ ở mục 0 vào nơi đọc tham số (đã tìm ở 3.2) và vào `args.sh` / `exps/*.json`. Mặc định = hành vi RSIAT gốc.
- Thêm `seed` kiểu số nguyên vào JSON và tùy chọn CLI `--seed`; ghi rõ nguồn seed cuối cùng sau merge. Seed này điều khiển khởi tạo mô hình, sampler/DataLoader và các nguồn ngẫu nhiên của Python/NumPy/PyTorch; cố định seed worker khi `num_worker > 0`. Thứ tự lớp vẫn luôn dùng seed 1993.
- Thêm `--num_worker` kiểu số nguyên không âm và hỗ trợ khóa JSON `"num_worker"`. Mặc định `8` để gần với loader train/test hiện tại; mọi `DataLoader` trong pipeline phải dùng giá trị cấu hình, kể cả loader ở `models/base.py` hiện dùng `4`. `--num_worker 0` phải chạy đồng bộ trong tiến trình chính để hỗ trợ debug. Đây là số worker trên mỗi loader, không phải tổng số worker của ứng dụng.
- Sửa quy tắc merge cấu hình: JSON cung cấp mặc định; chỉ tham số CLI được người dùng truyền rõ ràng mới override JSON. Không để giá trị mặc định của `argparse` vô tình ghi đè JSON.
- Chuẩn hóa `device` thành danh sách chỉ số GPU kiểu số nguyên trước khi khởi tạo learner/DataParallel; giữ CPU là trường hợp tường minh riêng. Kiểm tra số GPU và chỉ số GPU khả dụng, chọn `device[0]` làm device chính, và ghi danh sách GPU thực sự dùng cùng VRAM của từng GPU vào log.
- Thêm vào log: giá trị mọi cờ, số tham số huấn luyện được, số tham số của mô-đun lượng tử.

### Bước 4.1a: Hỗ trợ nhiều GPU và worker nạp dữ liệu

- Dùng `nn.DataParallel` làm đường triển khai ban đầu vì repo đã có nhánh này và quy trình continual-learning đang chạy trong một tiến trình. Ví dụ cấu hình `"device": ["0", "1"]`; vẫn giữ cấu hình một GPU. Trên Kaggle phải xác nhận cả hai T4 được runtime cấp trước khi chạy. Mỗi T4 giữ bản sao model và có VRAM riêng; đo peak memory từng GPU, không xem 2 GPU là một vùng VRAM gộp. Không chuyển sang DDP trong cùng thay đổi: DDP cần launcher, sampler phân tán, đồng bộ metric/trạng thái và xử lý riêng cho từng task; cân nhắc sau nếu cần hiệu năng/khả năng mở rộng cao hơn.
- Đảm bảo forward có gradient đi qua đối tượng DataParallel. Hiện `_network_module_ptr` được gán trước khi bọc và các lệnh gọi trực tiếp `extract_vector()`/`fc()` có thể bỏ qua phân tán dữ liệu/gradient. Điều chỉnh giao diện forward để trả về feature và logits cần cho loss, rồi tính loss từ tensor đã gather về GPU chính. Các optimizer parameter group có thể lấy tham số từ module gốc (`.module`), nhưng phép tính forward phải gọi wrapper; xử lý thuộc tính như `convnet`/`fc` đúng khi model đang được bọc.
- Rà soát trích xuất feature, eval và classifier calibration: nếu gọi phương thức custom trên `DataParallel` không được hỗ trợ hoặc gọi module gốc khiến chỉ một GPU chạy, dùng một forward được DataParallel hỗ trợ hoặc ghi rõ pha đó tuần tự có chủ đích. Không để wrapper lồng nhau; unwrap trước khi deepcopy/freeze hoặc backup classifier khi cần.
- Kiểm tra batch size có đủ lớn để chia cho số GPU; batch cuối nhỏ vẫn phải chạy đúng. Không khẳng định tăng tốc nếu chưa đo vì `DataParallel` có chi phí scatter/gather và batch nhỏ có thể không hiệu quả.
- Không đổi batch size mặc định, số bước optimizer, quy tắc lấy mẫu, thứ tự lớp hay giao thức đánh giá để làm kết quả nhiều GPU trông tương đương. Ghi cấu hình GPU và `num_worker` vào log.

### Bước 4.2: Cờ `--aligner qhybrid` (thay đổi A)

- Cài `QHybridAligner` có cùng giao diện với RAE hiện tại (đầu vào $f^{t-1}$, đầu ra cùng chiều).
- Khởi tạo $W_{\mathrm{up}}=0$ ngay đầu mỗi task $t>1$, giống RAE.
- Đảm bảo khi suy luận bộ căn chỉnh bị bỏ (assert không có tham số aligner trong đường eval).
- Kiểm tra: số tham số khi so với RAE (nên chỉnh cho tương đương để ablation công bằng).

### Bước 4.3: Cờ `--lambda_qrel` và `--kernel` (thay đổi B)

- Trong bước forward ở task $t>1$, lấy $f^{t-1}$ (stop-gradient, từ mô hình đóng băng) và $f^{t}$ trên cùng batch.
- Tính $K^{t-1}$, $K^{t}$ bằng kernel được chọn, rồi $\mathcal{L}_{\mathrm{qrel}} = \lVert K^{t}-\mathrm{sg}(K^{t-1})\rVert_F^{2}$.
- Cộng vào loss với trọng số $\beta(e)\,\lambda_{\mathrm{qrel}}$, dùng **cùng kiểu warm-up** với $\lambda_{\mathrm{RS}}(e)$.
- Cài đủ bốn kernel (`quantum`, `cosine`, `rbf`, `mlp`) cùng giao diện để chạy ablation quyết định.

### Bước 4.4: Cờ `--orth qweighted` (thay đổi C)

- Lấy prototype cũ $\hat p_c$ theo đúng cách $\mathcal{L}_{\mathrm{orth}}$ gốc đang lấy. Nếu gốc chưa bù drift thì thêm tùy chọn `--orth_use_drift_comp`.
- Với mỗi mẫu $i$: tính $F_{ic}$, tính $\alpha_{ic}=\mathrm{softmax}_c(F_{ic}/\tau)$ trên **toàn bộ lớp cũ**, sau đó chỉ cộng các hạng tử top-$k$ trong loss với hinge $\max(0,F_{ic}-\varepsilon)$. Không tính lại/chuẩn hóa softmax riêng trên top-$k$.
- Hyperparameter khởi điểm: $k=5$ (hoặc $\min(5, \lvert\mathcal{Y}_{1:t-1}\rvert)$), $\tau=0.1$, $\varepsilon$ chọn từ phân phối $F_{ic}$ trên dữ liệu train ở epoch đầu (ví dụ phân vị thứ 50); không dùng test set để chọn ngưỡng.
- Với task 2 có thể có ít lớp cũ: xử lý trường hợp $k >$ số lớp cũ.

### Bước 4.5: Cờ `--rs_kernel quantum` (thay đổi D, tùy chọn)

- Chỉ làm sau khi A–C ổn định. Thay phép cosine trong $S_{ij}$ của $\mathcal{L}_{\mathrm{RS}}$ bằng kernel lượng tử với cùng feature map.
- Cẩn thận: $\mathcal{L}_{\mathrm{RS}}$ dùng $1\pm S_{ij}$ với $S\in[-1,1]$, còn fidelity nằm trong $[0,1]$. Cần chuẩn hóa lại cho phù hợp (ví dụ dùng $2F-1$) và ghi rõ.

### Bước 4.6: Kiểm tra hồi quy cuối giai đoạn

- Tắt mọi cờ: kết quả phải khớp baseline (cùng seed và cùng thứ tự lớp seed 1993).
- Bật từng cờ riêng lẻ: không crash, không NaN, thời gian train tăng bao nhiêu thì ghi lại.
- Xác nhận pipeline nền và QR-RSIAT chạy trên một GPU và từ hai GPU trở lên. Trên nhiều GPU, kiểm tra batch thực sự được phân chia và gradient cập nhật được trọng số; chỉ kiểm tra không crash là chưa đủ. So sánh single-GPU/multi-GPU theo tolerance định trước vì thứ tự phép tính có thể tạo sai khác số học nhỏ.
- Kiểm tra `num_worker=0`, mặc định và một giá trị lớn hơn `0`; xác nhận tất cả loader nhận đúng giá trị. Chỉ benchmark một ma trận nhỏ để chọn số worker, không giả định tăng worker luôn nhanh hơn.
- Nếu runtime Kaggle không cấp hai GPU, chạy kiểm tra đơn vị cho parse/validation/chọn device, nhưng đánh dấu kiểm tra thực nghiệm multi-GPU là chưa xác minh.

---

## 7. Giai đoạn 5: Thực nghiệm

### 7.1. Ma trận ablation

| ID | Aligner | $\mathcal{L}_{\mathrm{qrel}}$ (kernel) | Orth | Ghi chú |
|---|---|---|---|---|
| E0 | RAE | tắt | plain | RSIAT gốc |
| E1 | không có | tắt | plain | bỏ projector |
| E2 | không có | tắt | plain, thêm L2 | mốc cho bỏ projector |
| E3 | qhybrid | tắt | plain | chỉ đổi aligner |
| E4 | RAE | cosine | plain | quan hệ + kernel cổ điển |
| E5 | RAE | quantum | plain | quan hệ + kernel lượng tử |
| E6 | RAE | tắt | qweighted | chỉ đổi orth |
| E7 | qhybrid | quantum | qweighted | **QR-RSIAT đầy đủ** |
| E8 | qhybrid | rbf / mlp | qweighted | kernel cổ điển, cùng số tham số |
| E9 | qhybrid, bỏ skip | quantum | qweighted | lặp lại thí nghiệm `wo_res` |

**So sánh quyết định:** E7 với E8 (và E5 với E4). Nếu kernel lượng tử không hơn cổ điển thì báo cáo trung thực như vậy.

### 7.2. Thiết lập

- Dataset/cấu hình: IN-R B0I20, IN-A B0I20, CUB B0I10, VTAB B0I10; thêm IN-R B0I10 (chuỗi dài) và IN-R B100I20 (large-base). Người dùng xác nhận các dataset và ViT-B/16-IN21K weights đã có trong Kaggle Input; xác minh mount/path trước khi chạy.
- Backbone: ViT-B/16-IN21K, giữ nguyên hyperparameter của RSIAT cho phần không đổi.
- **Ít nhất 3 seed** cho mỗi ô, báo cáo mean ± std. Mỗi seed chạy điều khiển khởi tạo mô hình, sampler/DataLoader và các nguồn ngẫu nhiên khác; thứ tự lớp luôn cố định theo seed 1993. Ghi rõ seed cấu hình, seed thứ tự lớp và thiết lập determinism; không giả định CUDA cho kết quả bitwise-identical.
- Ma trận chính: E0–E9 chỉ trên IN-R B0I20; E0 và E7 trên IN-A B0I20, CUB B0I10, VTAB B0I10, IN-R B0I10 và IN-R B100I20. Tổng cộng 20 ô cấu hình × 3 seed = **60 lượt huấn luyện chính**, chưa tính tối đa 20 lượt sweep, smoke test và xác minh hồi quy/GPU. Baseline IN-R/IN-A ở mục 4 chính là các ô E0 tương ứng, không chạy lặp.
- Sweep theo giai đoạn trên IN-R, tối đa **20 lượt sàng lọc tổng cộng**, mỗi cấu hình một seed: sàng lọc q/lq trước, sau đó lần lượt $\tau$, $k$, $\varepsilon$ và $\lambda_{\mathrm{qrel}}$. Ghi trước danh sách và số cấu hình ở từng chặng; không chạy tích Descartes đầy đủ. Dùng tiêu chí train/validation đã định trước, không dùng test để chọn cấu hình. Đóng băng cấu hình cuối trước các dataset khác, rồi chạy cấu hình đó đủ 3 seed. Báo cáo riêng sweep đơn-seed và kết quả xác nhận 3 seed.
- Phân bổ qua nhiều phiên Kaggle: mỗi lượt có ID cấu hình, seed, trạng thái (pending/running/done/failed), config đã merge, commit hash và đường dẫn checkpoint/log. Lưu checkpoint tối thiểu tại ranh giới task để tiếp tục an toàn; không ghi đè kết quả hoàn tất. Sau mỗi phiên, lưu sản phẩm cần tiếp tục trong `/kaggle/working`, Save Version/commit notebook và tải Output về nơi lưu bền trước khi kết thúc phiên. Trước phiên kế tiếp, khôi phục các Output/checkpoint đã tải hoặc gắn nguồn lưu bền trở lại làm input, rồi xác minh manifest và checksum trước khi tiếp tục.
- Trên mỗi phiên Kaggle, ghi runtime thực tế: Python, PyTorch, CUDA, driver, model/số GPU, VRAM từng GPU, RAM hệ thống, `device` IDs và `num_worker`. Không coi 30 GB RAM hoặc 2×T4 là khả dụng nếu phiên thực tế không cấp đủ.

### 7.3. Chỉ số cần ghi

- $\bar{\mathcal{A}}$, $\mathcal{A}_B$ theo từng task.
- **Độ trôi prototype:** khoảng cách giữa prototype lưu và prototype tính lại bằng adapter hiện tại trên dữ liệu task đó (khi có thể).
- Recall@1 trong không gian feature gốc (theo định nghĩa của RSIAT).
- Thời gian train mỗi task, thời gian suy luận mỗi ảnh, peak memory.
- Cấu hình GPU (`device` IDs) và `num_worker`; so sánh thời gian train/epoch khi thay đổi số worker nếu có tài nguyên.
- Số tham số huấn luyện được (tách riêng phần lượng tử).
- Gradient norm của loss lượng tử theo epoch (phát hiện concentration).

### 7.4. Cấu trúc lưu kết quả

```
results/
├── baseline.md
├── ablation_table.md        # bảng E0–E9, mean ± std
├── sweeps/
├── figures/
├── raw/                     # log thô, JSON theo từng lần chạy
└── checkpoints/             # checkpoint theo task để tiếp tục qua các phiên
```

Giữ bản có thể tải/lưu của `results/`, config đã merge và manifest tiến độ dưới `/kaggle/working`; sau mỗi phiên tạo Kaggle Save Version/commit notebook và tải Output. Không xem `/kaggle/working` đơn lẻ là lưu trữ bền qua phiên.

---

## 8. Quy tắc làm việc cho agent

1. **Đọc trước, sửa sau.** Không viết dòng code nào vào file gốc trước khi `REPO_MAP.md` hoàn thành.
2. **Diff nhỏ.** Ưu tiên thêm file mới và thêm điểm gắn (hook) tối thiểu vào file gốc.
3. **Hồi quy.** Sau mỗi commit chạm vào file gốc, chạy lại smoke test với cờ tắt.
4. **Không đoán.** Gặp chỗ mơ hồ (ví dụ: công thức trong paper khác code), ghi vào mục "Vấn đề mở" của `REPO_MAP.md` rồi hỏi người dùng.
5. **Không sửa dữ liệu hay giao thức đánh giá** để tăng điểm.
6. **Báo cáo thất bại.** Kết quả âm (lượng tử không hơn cổ điển, aligner làm giảm điểm) vẫn phải ghi đúng.
7. **Tiết kiệm tính toán.** Chạy thử trên dataset nhỏ và ít epoch trước, chỉ chạy đầy đủ khi smoke test và test đơn vị đều pass. Không vượt quá 20 lượt sàng lọc sweep đã thống nhất.
8. Mỗi giai đoạn kết thúc bằng một bản tóm tắt ngắn: đã làm gì, đã kiểm chứng gì, còn rủi ro nào.
9. Dùng máy cá nhân không GPU cho công việc CPU; mọi tuyên bố về huấn luyện GPU/multi-GPU phải dựa trên log Kaggle của phiên thực tế. Đo VRAM từng GPU riêng; `DataParallel` có thể chậm hơn với batch nhỏ, nên chỉ báo cáo hiệu năng theo số đo.
10. Trước khi phiên Kaggle kết thúc, lưu checkpoint/log/config/manifest sang Output có thể tải, rồi xác nhận dữ liệu đã được lưu bền. Không dựa vào trạng thái RAM hoặc filesystem tạm của phiên để tiếp tục.

---

## 9. Mốc bàn giao

| Mốc | Sản phẩm | Điều kiện |
|---|---|---|
| M0 | `REPO_MAP.md`, baseline smoke test | Giai đoạn 0–1 xong |
| M1 | Baseline tái lập trên IN-R, IN-A | Mỗi dataset ít nhất 3 seed; E0 được tái sử dụng trong ma trận; lệch paper trong ngưỡng |
| M2 | `quantum/` + test T1–T5 pass | Chưa gắn vào pipeline |
| M3 | Bốn cờ A–D hoạt động, hồi quy sạch | Cờ tắt = baseline |
| M4 | Bảng ablation E0–E9 và đối chứng đa dataset | IN-R E0–E9; E0/E7 trên cấu hình còn lại; 3 seed/ô, mean ± std; tối đa 20 lượt sweep |
| M5 | Báo cáo cuối: kết quả, hạn chế, rủi ro | Gồm cả kết quả âm |

---

## 10. Prompt khởi động (dán cho agent)

```text
Bạn đang làm việc trên bản sao của repo RSIAT (https://github.com/zjrzjrz/RSIAT).
Hãy đọc file AGENT_PLAN_QR-RSIAT.md ở thư mục gốc và làm theo đúng thứ tự giai đoạn.

Bắt đầu bằng Giai đoạn 0 và 1: KHÔNG sửa code gốc. Mục tiêu đầu tiên là tạo
REPO_MAP.md trả lời đầy đủ các câu hỏi ở mục 3 (kèm đường dẫn:dòng).

Dùng máy cá nhân không GPU cho phân tích và test CPU; chạy baseline/benchmark trên
Kaggle. Tôi đã xác nhận có ImageNet-R, ImageNet-A, CUB, VTAB và ViT-B/16-IN21K
weights trong Kaggle Input; kiểm tra input mount/path, không tự tải lại.
Môi trường dự kiến là 30 GB RAM và 2 GPU T4, nhưng phải ghi nhận và xác minh
runtime, VRAM riêng từng GPU, Python/PyTorch/CUDA/driver mỗi phiên.

Thêm seed cấu hình/CLI để điều khiển khởi tạo mô hình, sampler/DataLoader và các
nguồn ngẫu nhiên khác; giữ cố định thứ tự lớp theo seed 1993. Lưu checkpoint theo
task, log, config đã merge và manifest dưới /kaggle/working; sau mỗi phiên, Save
Version/commit notebook và tải Output để bảo toàn kết quả.

Chạy E0-E9 trên IN-R B0I20; chạy E0/E7 trên IN-A, CUB, VTAB, IN-R B0I10 và
IN-R B100I20; mỗi ô 3 seed. Sweep theo giai đoạn trên IN-R, tối đa 20 lượt sàng
lọc một-seed tổng cộng, sau đó chạy cấu hình đã chốt đủ 3 seed. Với qweighted,
softmax trên toàn bộ lớp cũ rồi chỉ lấy top-k trong loss.

Dừng và báo cáo nếu dataset/input thực tế thiếu hoặc không đọc được, runtime
không tương thích, code khác đáng kể với giả định trong tài liệu, hoặc baseline
không tái lập được. Không báo multi-GPU đã kiểm chứng nếu phiên không cấp đủ hai
GPU. Sau mỗi giai đoạn, tóm tắt những gì đã làm và đã kiểm chứng.
```

---

## Phụ lục A: Tóm tắt công thức để agent đối chiếu

Kernel và alignment quan hệ:

$$
K_{ij} = \left\lvert \langle \psi(z_i)\mid\psi(z_j)\rangle \right\rvert^{2},
\qquad
\mathcal{L}_{\mathrm{qrel}} = \left\lVert K^{t}-\mathrm{sg}\!\left(K^{t-1}\right)\right\rVert_F^{2}
$$

Bộ căn chỉnh đồng nhất khi khởi tạo:

$$
P^{t}(f) = f + W_{\mathrm{up}}\,\mathrm{Readout}\!\left(\mathrm{PQC}(W_{\mathrm{down}}f)\right),
\qquad W_{\mathrm{up}}^{(0)}=0
$$

Trực giao chọn lọc:

$$
F_{ic}=\left\lvert\langle\psi(\hat u_i)\mid\psi(\hat p_c)\rangle\right\rvert^{2},\quad
\alpha_{ic}=\mathrm{softmax}_c\!\left(F_{ic}/\tau\right),\quad
\mathcal{L}_{\mathrm{orth}}^{q}=\frac{1}{\lvert\mathcal{B}\rvert}\sum_{i}\sum_{c\in\mathrm{top}\text{-}k_i}\alpha_{ic}\max(0,F_{ic}-\varepsilon)
$$

Loss tổng:

$$
\mathcal{L}^{(t)}=\mathcal{L}_{\cos}^{(t)}+\beta(e)\left(\mathcal{L}_{\mathrm{qrel}}+\mathcal{L}_{\mathrm{align}}\right)+\gamma(e)\,\mathcal{L}_{\mathrm{orth}}^{q},
\qquad
\beta(e)=\beta\min\!\left(1,\tfrac{e}{E_w}\right)
$$

## Phụ lục B: Checklist cuối trước khi báo cáo

- [ ] `REPO_MAP.md` đầy đủ, có mục "Vấn đề mở".
- [ ] Baseline tái lập, có số liệu.
- [ ] Test T1–T5 pass, T6–T7 đã ghi lại.
- [ ] Mọi cờ tắt cho kết quả giống baseline (cùng seed).
- [ ] Đường suy luận không chứa aligner hay mạch lượng tử (có test).
- [ ] Bảng ablation có mean ± std từ $\ge 3$ seed.
- [ ] E0–E9 chỉ chạy trên IN-R B0I20; E0/E7 trên các dataset/cấu hình còn lại theo mục 7.2.
- [ ] Thứ tự lớp cố định seed 1993; seed config tái lập các nguồn ngẫu nhiên còn lại.
- [ ] Sweep không quá 20 lượt sàng lọc; cấu hình cuối được xác nhận bằng 3 seed.
- [ ] Checkpoint, config đã merge, log và manifest được lưu bền sau mỗi phiên Kaggle.
- [ ] Phiên bản runtime và VRAM từng GPU được ghi lại; RAM hệ thống không bị nhầm với VRAM GPU.
- [ ] So sánh công bằng kernel lượng tử với cosine/RBF/MLP cùng số tham số.
- [ ] Kết quả âm được báo cáo.
- [ ] Không có số liệu nào không đến từ lần chạy thật.
