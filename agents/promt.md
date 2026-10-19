# PROMT.md — Bộ prompt khắc phục và tối ưu QR-RSIAT

Tài liệu gồm **một prompt ngữ cảnh (P00)** và **18 prompt (P01–P18)**. Mỗi prompt xử lý một vấn đề liên quan tới logic code, kèm bằng chứng, cơ sở khoa học, đề xuất khắc phục, một khối prompt có thể dán cho trợ lý lập trình, và tiêu chí nghiệm thu. Mục tiêu: **rút ngắn thời gian huấn luyện** và **tăng hiệu quả/độ tin cậy kết luận** của QR-RSIAT.

## 0. Cách dùng

1. Dán **P00** một lần ở đầu mỗi phiên làm việc.
2. Dán từng prompt (P01…P18) theo thứ tự ở mục 3. Không gộp nhiều prompt vào một lần.
3. Nhãn mức chắc chắn dùng xuyên suốt:
   - **[Code]**: thấy trực tiếp trong mã nguồn đã cung cấp.
   - **[Log]**: thấy trong log hoặc cấu hình của lần chạy `imageneta`, seed 1992.
   - **[Đo]**: kiểm chứng bằng mô phỏng numpy (không phải torch, không phải GPU).
   - **[Suy luận]**: chưa có bằng chứng trực tiếp, cần đo (P01) trước khi tin.
4. Mọi con số "ước tính" về thời gian tiết kiệm phải được xác nhận bằng profiler trên Kaggle. Không dùng chúng làm cam kết.
5. Tài liệu tham khảo ở cuối được chia hai nhóm: đã xác minh bằng tìm kiếm và từ trí nhớ (cần kiểm tra lại trước khi trích dẫn).

---

## 1. Bằng chứng nền

### 1.1 Cấu hình lần chạy [Log]

| Mục | Giá trị |
|---|---|
| Bộ dữ liệu / giao thức | ImageNet-A, 10 task × 20 lớp (`init_cls=20`, `increment=20`) |
| Mạng | ViT-B/16 IN21k + adapter (`ffn_num=64`); backbone đóng băng, 1,189,632 tham số huấn luyện |
| Optimizer | `sgd` (momentum 0.9), `init_lr=0.01`, `weight_decay=5.93e-7`, `batch_size=48`, AMP fp16 |
| Epoch | `init_epochs=20`, `inc_epochs=20`, `ca_epochs=10`, `warmup_epoch=3` |
| CosFace | `scale=30`, `margin=0.3` |
| QR-RSIAT | `aligner=qhybrid`, `kernel=quantum`, `rs_kernel=quantum`, `orth=qweighted`, `n_qubits=8`, `quantum_layers=2`, `cycle=True`, `quantum_centering=True` |
| Hệ số loss | `lambda_rs=1`, `rs_margin=1`, `beta=3`, `gamma=6`, `lambda_qrel=0.4`, `orth_top_k=2`, `orth_temperature=0.1`, `orth_epsilon=0.5` |
| Khác | `ssca=True`, `ca=True`, `pair_gather=False`, một GPU, một seed |

### 1.2 Kết quả [Log]

Đường cong top-1: 88.18, 85.46, 80.86, 79.30, 77.34, 74.91, 71.59, 69.14, 67.78, 66.67. **Average Accuracy = 76.12.** Độ chính xác lớp mới giảm mạnh ở các task ít dữ liệu: task 5 (13 ảnh/lớp) còn 55.88%, task 6 (18 ảnh/lớp) còn 42.70%.

### 1.3 Thời gian [Log]

| Task | Ảnh | Bước/epoch | Huấn luyện (giây/epoch) | Sau huấn luyện (giây) |
|---|---|---|---|---|
| 0 | 1253 | 27 | 16.8 | 34 |
| 1 | 1000 | 21 | 24.4 | 151 |
| 2 | 1000 | 21 | 27.9 | 199 |
| 3 | 610 | 13 | 24.0 | 220 |
| 4 | 426 | 9 | 22.5 | 240 |
| 5 | 263 | 6 | 20.8 | 256 |
| 6 | 351 | 8 | 23.5 | 284 |
| 7 | 321 | 7 | 24.1 | 306 |
| 8 | 391 | 9 | 26.7 | 336 |
| 9 | 385 | 9 | 28.0 | 363 |

- Huấn luyện tổng ≈ **79.6 phút**; giai đoạn sau huấn luyện (thống kê lớp, drift, CA, đánh giá) ≈ **39.8 phút**; toàn bộ ≈ 121 phút.
- Thời gian mỗi epoch **gần như không phụ thuộc số bước**: task 5 chỉ 6 bước/epoch nhưng vẫn mất 20.8 giây. Suy ra phần cố định theo epoch ≥ ~18 giây. `_compute_accuracy` chạy toàn bộ tập test mỗi epoch bằng fp32 không autocast (`RSIAT_adapter.py:700`, `base.py:218-232`) là nghi phạm chính [Code + Log, mức đóng góp cần profile].
- Giai đoạn sau huấn luyện tăng gần tuyến tính theo số lớp (151 → 363 giây), phù hợp với vòng lặp theo lớp trong CA (`base.py:117-128`).

### 1.4 Số tham số khớp với phân tích mã [Log + Code]

- Aligner: `down` 768×16+16 = 12,304, `theta` 16, `up` 16×768 = 12,288 → **24,608**.
- Mỗi kernel lượng tử: 12,304 + 16 = **12,320**.
- Task 0: 61,568 = 24,608 + 3×12,320 (aligner + ba kernel). Task ≥ 1: 49,248 = 24,608 + 2×12,320 (không còn `rs_kernel`).

### 1.5 Chế độ hoạt động của kernel [Đo]

Mô phỏng q=8, L=2, `cycle=True`, chiếu Gaussian i.i.d. theo `σ_u ≈ ‖z−c‖/√(3·768)` (khởi tạo `nn.Linear` mặc định). `radius = 1.5741637924924812` in ra sau task 0 là RMS độ lệch chuẩn theo từng chiều **trong lớp**, suy ra `‖z − μ_lớp‖ ≈ 43.6`.

| Kernel | Feature vào kernel | Kết quả mô phỏng |
|---|---|---|
| RS (base task) | L2-normalize (`RSIAT_adapter.py:781`), đã center | `σ_u ≈ 0.006–0.02`, F ngoài đường chéo ≈ **0.97–0.999** cho mọi cặp |
| qrel | feature thô | F cặp cùng lớp: trung bình 0.0036 (p99 0.039); cặp khác lớp: 0.0036 (p99 0.046) → **K ≈ I, không phân biệt lớp** |
| qorth, G của top-1 prototype | feature thô | P(>0.2) ≤ 2%, P(>0.3) ≤ 0.7%, P(>0.5) = 0 trong 300 mẫu (C_old = 20/100/180) |

### 1.6 Giới hạn của bằng chứng

- Mô phỏng dùng chiếu Gaussian i.i.d. ở thời điểm khởi tạo; feature thật có cấu trúc tương quan nên F thực có thể cao hơn. Chưa đo `‖W_q‖` thật.
- Chỉ một seed, không có baseline `aligner=rae` cùng điều kiện, nên **chưa thể quy kết quả 76.12 cho phần lượng tử**.
- Chưa đọc `qrsiat/stats/drift.py`, `qrsiat/stats/accumulators.py`, `learner_mixin.py`, `base.py` dòng 180–215.

---

## 2. Tổng quan các prompt

Cột "Tác động": **T** = thời gian, **H** = hiệu quả huấn luyện/độ chính xác, **K** = tính đúng đắn khoa học của kết luận.

| ID | Vấn đề | Loại | Tác động | Ưu tiên | Phụ thuộc |
|---|---|---|---|---|---|
| P01 | Không có log từng thành phần loss và thống kê kernel | Code | K | Cao nhất | — |
| P02 | Không có baseline, một seed, không quy được kết quả | Log | K | Cao nhất | P01 |
| P03 | RS (norm 1) và qrel/qorth (thô) lệch ~100× về thang; kernel K≈I | Code + Đo | H, K | Cao | P01 |
| P04 | `rs_margin=1` làm `L_neg ≡ 0`; margin cố định không bám phân bố F | Code + Log | H | Cao | P01 |
| P05 | Ba kernel riêng, teacher kernel ngẫu nhiên, kernel hấp thụ loss | Code + Log | H, K | Cao | P03 |
| P06 | qorth: softmax không detach, aligner nhận gradient từ qorth | Code + Đo | K | Trung bình | P05 |
| P07 | `θ` lớp cuối và CNOT cuối vô tác dụng với fidelity; readout bị bỏ | Đo | T, K | Trung bình | — |
| P08 | Centering không nhất quán giữa các đường | Code | H | Trung bình | P03 |
| P09 | Nhánh `adam` bỏ qua param_groups; nhóm lượng tử dùng LR của network | Code | H, K | Trung bình | P05 |
| P10 | Warm-up không cân bằng giữa `rae` và `qhybrid`; epoch 0 không có regularizer | Code | H, K | Trung bình | P02 |
| P11 | `L_align` chia thêm 768; aligner học chậm, không biết có giá trị không | Code | H, K | Trung bình | P01 |
| P12 | `sigma=4.0` cố định cho ssca, khoảng cách thật cỡ 40+ | Code + Log | H | **Cao** | P01 |
| P13 | Aligner không tham gia cập nhật prototype | Code | H | Thấp (thực nghiệm) | P11, P12 |
| P14 | Covariance lớp cũ không cập nhật; hạng thấp ở lớp ít dữ liệu | Code + Log | H | Cao | P12 |
| P15 | CA tính lại eigh/Cholesky mỗi epoch, đánh giá test mỗi epoch | Code + Log | T, H | **Cao** | — |
| P16 | Đánh giá fp32 mỗi epoch; DataLoader theo từng lớp; sync mỗi bước | Code + Log | T | **Cao nhất về thời gian** | — |
| P17 | Simulator: mask boolean gây đồng bộ GPU, nhiều thao tác nhỏ | Code + Đo | T | Trung bình | P07 |
| P18 | Mã chết, tài liệu lệch mã, `probe` đổi batch, tái lập | Code | K | Thấp | — |

## 3. Thứ tự thực hiện

1. **Giai đoạn 0 (đo):** P01, khung của P02. Chưa sửa gì liên quan đến khoa học.
2. **Giai đoạn 1 (rút ngắn thời gian, không đổi kết quả):** P16, P15 (phần cache và tần suất đánh giá), P17, P18.
3. **Giai đoạn 2 (sửa logic):** P04, P03, P05, P06, P07, P08, P10, P11, P09.
4. **Giai đoạn 3 (cải thiện độ chính xác ngoài nhánh lượng tử):** P12, P14, P13.
5. **Giai đoạn 4:** chạy ma trận ablation đầy đủ của P02 với ≥ 3 seed, rồi **quyết định giữ hoặc bỏ từng nhánh lượng tử**.

Quy tắc quyết định: nhánh nào (qrel, qorth, RS, aligner, mạch lượng tử so với kernel cổ điển) không cho khác biệt có ý nghĩa thống kê so với bỏ nó thì bỏ, vì nó chỉ tốn thời gian.

---

## P00 — Prompt ngữ cảnh (dán đầu mỗi phiên)

```text
Bạn đang làm việc trên repo QR-RSIAT: học tăng dần theo lớp (class-incremental learning)
dựa trên RSIAT. Mạng: ViT-B/16 (IN21k) + adapter. Backbone đóng băng; chỉ adapter,
classifier cosine (CosFace) và các module lượng tử mô phỏng (statevector thực, cổng RY và
CNOT) được huấn luyện. Nhánh lượng tử chỉ dùng khi huấn luyện; suy luận chỉ đi qua
feature -> classifier.

Cấu trúc chính: main.py, trainer.py; models/RSIAT_adapter.py (learner, optimizer, vòng
huấn luyện, RS_Loss); models/base.py (thống kê lớp, CA, drift cổ điển);
qrsiat/training/step_module.py (forward chung và các loss qrel/qorth);
qrsiat/quantum/{feature_map,simulator,kernels,aligner,losses}.py;
qrsiat/training/schedule.py; qrsiat/stats/{accumulators,drift}.py.

Ràng buộc bắt buộc:
1. KHÔNG làm đổi hành vi mặc định. Mọi thay đổi hành vi nằm sau cờ cấu hình mới, mặc định
   giữ đúng hành vi hiện tại (aligner mặc định vẫn là "rae"). Cờ mới phải được validate
   sớm trong qrsiat/config.
2. Máy cục bộ không có CUDA: KHÔNG chạy huấn luyện. Viết unit test chạy được trên CPU
   (torch CPU hoặc numpy) cho từng thay đổi. Việc chạy trên GPU do người dùng thực hiện.
3. Giữ tính xác định (seed), checkpoint nguyên tử, và khả năng resume. Nếu thay đổi
   tham số hoặc buffer được lưu, phải cập nhật CheckpointManager và kiểm tra tương thích.
4. Mô phỏng lượng tử luôn chạy fp32, ngoài autocast.
5. Không thêm phụ thuộc mới (Kaggle không cài lại torch/torchvision).
6. Mỗi lần giao việc kết thúc bằng: (a) danh sách file đã sửa, (b) test đã thêm và cách
   chạy, (c) rủi ro và điểm cần người dùng xác nhận, (d) KHÔNG tuyên bố "đã cải thiện độ
   chính xác" khi chưa có kết quả chạy.

Bằng chứng nền (lần chạy seed 1992): Average Accuracy 76.12; thời gian huấn luyện ~80
phút, sau huấn luyện ~40 phút; radius feature trong lớp = 1.574 (||z - mu_lop|| ~ 43.6);
RS_Loss L2-normalize feature trước kernel, còn qrel/qorth dùng feature thô.
```

---

## P01 — Log chẩn đoán từng thành phần loss và thống kê kernel (tiên quyết)

**Ưu tiên:** cao nhất · **Tác động:** K · **Phụ thuộc:** không · **Chi phí:** thấp

**Vấn đề (logic code).** `_compute_rt_loss` (`RSIAT_adapter.py:766-769`) cộng `loss_rs + loss_rel + loss_align·beta + loss_orth` thành một số `loss_rt`, và log chỉ có `Losses_rt`. `StepModule` đã trả về bốn thành phần riêng nhưng chúng bị gộp. Không có thống kê F/G, norm feature, `‖W_q‖`, `‖W_up‖`, hay độ trôi feature. Vì vậy không thể biết nhánh nào đang hoạt động. Ngoài ra `_init_train` gọi `.item()` ba lần và `.cpu()` một lần mỗi bước (`:687-691`), mỗi lần là một điểm đồng bộ GPU.

**Bằng chứng.** [Log] `Losses_rt = 0.010` ở task 0 (chỉ có `L_pos`); 0.12–0.25 ở các task sau, không tách được. [Đo] dự đoán `L_qrel ~ 1e-6`, qorth không kích hoạt: **chỉ log mới xác nhận hoặc bác bỏ**.

**Cơ sở khoa học.** Nguyên tắc "đo trước khi sửa" cho nhánh phụ trợ chỉ dùng lúc huấn luyện: cần đo trực tiếp tỉ lệ hinge đang hoạt động và độ lớn gradient. Kernel-target alignment (Cristianini và cộng sự, 2001) và CKA có tâm (Cortes và cộng sự, 2012) cho một số vô hướng đo K phản ánh nhãn tốt đến đâu. Cosine giữa các gradient (Yu và cộng sự, 2020, PCGrad) là chẩn đoán chuẩn cho các mục tiêu cạnh tranh.

**Đề xuất khắc phục.**
1. Trả về và log riêng `loss_rs`, `loss_rel`, `loss_align`, `loss_orth` mỗi epoch (trung bình trên các bước).
2. Cờ `--diagnostics` (mặc định tắt) ghi `diagnostics.jsonl` (chỉ rank 0), ở epoch 1 và epoch cuối mỗi task, trên một batch cố định không augmentation:
   - `s` = RMS theo chiều của feature đã center (thô và đã normalize), `σ_u`, tỉ lệ `|tanh u| > 0.99`.
   - Phân vị 5/50/95/99 của F cho cặp cùng lớp và khác lớp (kernel RS và kernel qrel), CKA có tâm giữa K và ma trận nhãn.
   - Với qorth: phân bố G của top-k và `P(G_top1 > ε)`.
   - `‖W_q‖_F`, `‖θ‖`, `‖W_up‖_F`, khoảng cách trung bình `‖aligned_prototype − prototype‖`.
   - Độ trôi `‖f_t − f_{t−1}‖ / ‖f_{t−1}‖` trên ảnh không augmentation.
   - Tỉ lệ norm gradient trên tham số adapter của từng loss so với `L_cos`, và cosine giữa các gradient (mỗi k bước, bằng `autograd.grad`).
3. Gom bốn lần `.item()`/`.cpu()` mỗi bước thành tích lũy trên GPU, chuyển về CPU một lần cuối epoch.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P01: thêm chẩn đoán mà KHÔNG đổi hành vi huấn luyện mặc định.

1. models/RSIAT_adapter.py: _compute_rt_loss hiện trả (logits, loss_cls, loss_rt) với
   loss_rt gộp bốn thành phần. Thêm cách lấy riêng loss_rs, loss_rel, loss_align,
   loss_orth (ví dụ lưu self._last_components, đã detach). Trong _init_train, log từng
   thành phần trung bình theo epoch cạnh Losses_rt (giữ nguyên định dạng cũ để không phá
   công cụ phân tích log hiện có; thêm một dòng mới).
2. Thay các lệnh .item()/.cpu() trong vòng lặp bước bằng tích lũy tensor trên GPU; chỉ
   gọi .item() một lần cuối epoch. Kết quả số phải giống hệt trước đó.
3. Thêm cờ cấu hình diagnostics (mặc định False). Khi bật, ở epoch đầu và epoch cuối của
   mỗi task, dùng một batch cố định (ảnh không augmentation) để tính và ghi vào
   <output_dir>/<dataset>/seed_<seed>/diagnostics.jsonl (chỉ rank 0): các thống kê trong
   danh sách ở mục "Đề xuất" của P01 (RMS feature, sigma_u, tỉ lệ bão hòa tanh, phân vị F
   cùng lớp/khác lớp, CKA có tâm, phân bố G và P(G_top1 > epsilon), ||W_q||, ||theta||,
   ||W_up||, khoảng cách prototype đã căn chỉnh, độ trôi feature, tỉ lệ norm gradient và
   cosine gradient của từng loss so với loss phân loại trên tham số adapter).
   Dùng Phụ lục A3 làm điểm xuất phát cho hàm thống kê kernel.
4. Test (CPU): (a) cờ tắt thì mọi giá trị loss và gradient giống hệt phiên bản cũ trên một
   batch giả với seed cố định; (b) kernel_diagnostics trả về đúng khoá và F cùng lớp > F
   khác lớp trên dữ liệu tổng hợp có cấu trúc lớp rõ; (c) tích lũy GPU cho cùng tổng với
   cách .item() từng bước.
5. Chi phí khi bật diagnostics phải < 5% thời gian epoch (đo bằng thời gian tường).
Báo cáo theo mẫu của P00.
```

**Nghiệm thu.** Cờ tắt: kết quả bit-wise như cũ. Cờ bật: có đủ các trường; thời gian tăng < 5%. Sau một lần chạy, trả lời được: `L_qrel` thực tế bằng bao nhiêu? `P(G_top1 > ε)` bằng bao nhiêu? F cùng lớp có khác F khác lớp trong kernel qrel không?

---

## P02 — Giao thức baseline và ablation có ý nghĩa thống kê

**Ưu tiên:** cao nhất · **Tác động:** K · **Phụ thuộc:** P01 · **Chi phí:** trung bình (thời gian GPU)

**Vấn đề.** [Log] Chỉ có một lần chạy một seed (76.12), không có baseline `aligner=rae` cùng điều kiện, không có ablation từng thành phần. Không thể quy kết quả cho phần lượng tử. Dự đoán có thể bác bỏ [Đo + Suy luận]: tắt `lambda_qrel` và `gamma` (và RS) không làm đổi kết quả có ý nghĩa, vì ở thang hiện tại qrel/qorth gần như không hoạt động.

**Cơ sở khoa học.** Bowles, Ahmed và Schuld (2024) cho thấy kết quả benchmark mô hình học máy lượng tử rất nhạy với thiết kế thí nghiệm, baseline cổ điển "out-of-the-box" thường thắng, và việc bỏ rối thường không làm giảm (có khi tăng) hiệu năng. Bouthillier và cộng sự (2021) chỉ ra phương sai do seed, khởi tạo và thứ tự dữ liệu có thể lớn hơn khác biệt giữa các phương pháp, nên phải báo cáo phương sai và kiểm định cặp.

**Đề xuất khắc phục.** Ma trận cấu hình (ghi tên cố định, mỗi cấu hình là một file JSON override):

| Tên | Mục đích |
|---|---|
| B0 | `aligner=rae` (RSIAT gốc), cùng lịch warm-up (xem P10) |
| B1 | `qhybrid`, chỉ `L_align` (`lambda_qrel=0`, `gamma=0`, `lambda_rs=0`) |
| B2 / B3 / B4 | B1 cộng lần lượt qrel / qorth / RS |
| B5 | Cấu hình đầy đủ hiện tại |
| C1 | qrel và RS dùng `kernel=cosine` đã center |
| C2 | qrel và RS dùng `kernel=rbf` |
| C3 | Kernel cổ điển trên cùng 16 góc (xem P07) |
| Q1 / Q2 | L=1 (công thức đóng) / bỏ CNOT |

Số seed: tối thiểu 5 (1992 và 4 seed khác) cho kết luận cuối; giai đoạn đầu dùng 3 seed cho B0, B1, B5, C1. Chỉ số: Average Accuracy, Last Accuracy, Forgetting (độ giảm so với tốt nhất của từng task), độ chính xác lớp cũ và lớp mới. Báo cáo trung bình ± độ lệch chuẩn, khoảng tin cậy bootstrap, kiểm định cặp theo seed.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P02: tạo khung chạy ablation và tổng hợp thống kê.

1. Tạo thư mục configs/experiments/ablation_v2/ với một JSON override cho mỗi tên trong
   bảng (B0..B5, C1, C2, Q1; C3 và Q2 sẽ được thêm sau P07). Mỗi JSON chỉ chứa khóa ghi
   đè so với cấu hình cơ sở adapter_imageneta.json. Thêm kiểm tra không có khóa lạ.
2. Viết scripts/aggregate_results.py: đọc kết quả của mọi (cấu hình, seed) trong
   <output_dir>, tính Average Accuracy, Last Accuracy, Forgetting, độ chính xác lớp cũ
   và lớp mới theo task; xuất bảng markdown với trung bình +- độ lệch chuẩn và khoảng tin
   cậy bootstrap 95%; kiểm định cặp (theo seed) mỗi cấu hình so với B0 và so với B1.
3. Viết scripts/queue_ablation.py dùng chế độ queue có sẵn của qrsiat/runner để lập lịch
   (cấu hình x seed), hỗ trợ --seeds và --configs, tôn trọng --time_budget_h.
4. Test (CPU): aggregate_results chạy đúng trên dữ liệu kết quả giả; kiểm định cho kết
   quả mong đợi khi hai nhóm giống nhau (p lớn) và khi khác rõ (p nhỏ).
5. KHÔNG chạy huấn luyện. In ra danh sách lệnh để người dùng chạy trên Kaggle và ước
   lượng thời gian GPU tổng (số run x thời gian mỗi run đo được từ log).
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Có bảng tổng hợp tự động. **Quy tắc quyết định** (ghi vào báo cáo): nhánh nào không khác biệt có ý nghĩa so với phiên bản thiếu nó thì loại; nếu C1 (cosine đã center) ngang hoặc hơn B5 thì phần lượng tử không có chỗ đứng trong kết luận.

---

## P03 — Thống nhất thang đo feature và hiệu chỉnh băng thông kernel

**Ưu tiên:** cao · **Tác động:** H, K · **Phụ thuộc:** P01 · **Chi phí:** trung bình

**Vấn đề.**
- [Code] `RS_Loss.forward` L2-normalize feature trước kernel (`RSIAT_adapter.py:781`), còn qrel và qorth truyền feature thô vào kernel (`step_module.py`). Cùng một kernel lượng tử nhìn hai thang cách nhau ~100×.
- [Log + Đo] Ở RS, `σ_u ≈ 0.006–0.02` nên F ≈ 0.97–0.999 cho mọi cặp (vùng phẳng, gradient theo góc tỉ lệ với Δα nên rất nhỏ). Ở qrel/qorth, `‖z−μ‖ ≈ 43.6` làm F ≈ 0.0036 cho mọi cặp, cùng lớp lẫn khác lớp: K ≈ I, không mang quan hệ nào; hinge `[G − ε]₊` hầu như không kích hoạt.
- [Code] `QuantumFeatureMap` không có tham số băng thông; `nn.Linear` khởi tạo mặc định và `tanh` cố định.

**Cơ sở khoa học.**
- Shaydulin và Wild (2022): băng thông kernel quyết định mô hình underfit hay overfit; không tối ưu băng thông thì giá trị kernel suy giảm theo số qubit và hiệu năng kém.
- Thanasilp, Wang, Cerezo và Holmes (2024): các nguồn gây concentration gồm expressivity của embedding, phép đo toàn cục, rối và nhiễu; huấn luyện embedding tham số hóa bằng kernel alignment cũng chịu concentration.
- Huang và cộng sự (2021): fidelity kernel với chiều hiệu dụng lớn coi mọi dữ liệu là xa nhau, nên K gần ma trận đơn vị và mô hình cổ điển cạnh tranh được; đề xuất projected kernel.
- Heuristic trung vị cho băng thông (Garreau, Jitkrittum, Kanagawa, 2017) và khởi tạo phụ thuộc dữ liệu (Salimans và Kingma, 2016; Mishkin và Matas, 2016) là thực hành chuẩn.

**Đề xuất khắc phục.**
1. Một chuẩn hóa đầu vào duy nhất cho cả RS, qrel, qorth (LayerNorm không affine, hoặc L2-normalize nhân hằng số), tùy chọn qua cờ.
2. Thêm buffer `bandwidth` γ vào `QuantumFeatureMap`: `α = π·tanh(γ·u) + θ`.
3. Hiệu chỉnh γ ở đầu mỗi task (và base) trên 1–2 batch không augmentation: chọn γ trên lưới log sao cho trung vị F của cặp khác lớp (RS) hoặc của mọi cặp (qrel, qorth) bằng một mục tiêu cố định (ví dụ 0.1). Teacher snapshot của qrel dùng đúng γ của student trong task đó. Xem Phụ lục A2.
4. Ghi γ, trung vị F và CKA vào diagnostics (P01).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P03: thống nhất thang đo và hiệu chỉnh băng thông kernel lượng tử, sau cờ.

1. qrsiat/quantum/feature_map.py: thêm buffer "bandwidth" (mặc định 1.0, tương thích
   ngược) và dùng angles = pi * tanh(bandwidth * u). Thêm tùy chọn input_norm trong
   {"none", "layernorm", "l2"} (mặc định "none"); với "l2" nhân thêm sqrt(D). Cập nhật
   checkpoint để lưu buffer bandwidth.
2. qrsiat/quantum/kernels.py và step_module.py: dùng CÙNG input_norm cho RS, qrel và
   qorth khi cờ bật (hiện RS normalize trong RS_Loss còn qrel/qorth thì không). Giữ hành
   vi cũ khi cờ tắt.
3. Thêm hàm calibrate_bandwidth (xem Phụ lục A2): lưới log gamma, chọn gamma sao cho
   trung vị F (cặp khác lớp nếu có nhãn, nếu không thì mọi cặp ngoài đường chéo) gần mục
   tiêu kernel_target_median (mặc định 0.1). Gọi ở đầu mỗi task trên 1-2 batch không
   augmentation, dưới torch.no_grad() và full precision. Cờ kernel_calibration (mặc định
   false). Ghi gamma và trung vị F vào log.
4. Với qrel, snapshot teacher phải sao chép cả gamma. Với qorth, kernel dùng gamma đã
   hiệu chỉnh trên feature và prototype (cùng thang).
5. Test (CPU): (a) bandwidth=1, input_norm none cho kết quả giống hệt bản cũ; (b) sau
   calibrate, trung vị F nằm trong +-0.03 mục tiêu trên dữ liệu Gaussian tổng hợp ở các
   thang norm 1, 10, 50; (c) F cùng lớp > F khác lớp sau calibrate trên dữ liệu có cụm.
Báo cáo theo mẫu P00, nêu rõ rủi ro đổi hành vi so với cấu hình hiện tại.
```

**Nghiệm thu.** Sau hiệu chỉnh, phân vị F của cặp khác lớp không còn ≈ 0.0036 hoặc ≈ 0.99 mà nằm gần mục tiêu; CKA có tâm tăng; `P(G_top1 > ε)` ở mức có nghĩa (không ≈ 0). **Chỉ khi đó** mới đáng chạy ablation qrel/qorth.

---

## P04 — Kiểm tra `rs_margin` và đặt margin theo phân bố F

**Ưu tiên:** cao · **Tác động:** H · **Phụ thuộc:** P01 · **Chi phí:** thấp

**Vấn đề.** [Log + Code] Cấu hình đặt `rs_margin: 1`. F đã bị `clamp(0, 1)` (`fidelity_kernel`) nên `relu(F − 1) = 0` với mọi cặp: `L_neg ≡ 0` tuyệt đối, RS chỉ còn lực hút `L_pos` (`Losses_rt = 0.010` ở task 0 chính là `L_pos`). Tài liệu yêu cầu margin `< 1` nhưng giá trị đúng bằng 1 không bị chặn. Ngoài ra `[F − m]₊` với `m` cố định không có nghĩa khi phân bố F thay đổi mạnh (P03).

**Cơ sở khoa học.** Margin của hinge/triplet phải tương ứng thang tương đồng; với thang bão hòa, margin cố định hoặc vô hiệu hoặc luôn kích hoạt (xem CosFace, Wang và cộng sự 2018, về vai trò của margin gắn với thang cosine; FaceNet, Schroff và cộng sự 2015, về chọn âm bán khó). Lực hút một mình có nghiệm tầm thường (mọi mẫu cùng một state), nên cần lực đẩy hoạt động thật sự.

**Đề xuất khắc phục.**
1. Validation chặt: khi `rs_kernel=quantum`, yêu cầu `0 ≤ rs_margin < 1` (loại giá trị bằng 1 và lớn hơn 1).
2. Tùy chọn `rs_margin_mode ∈ {fixed, quantile}`: ở chế độ `quantile`, margin đầu mỗi epoch (EMA) bằng phân vị `q` (mặc định 0.9) của F cặp âm, để luôn còn một tỉ lệ cặp âm kích hoạt.
3. Tùy chọn `rs_pos_target < 1` thay cho `[1 − F]₊` để không ép co về một điểm.
4. Log tỉ lệ cặp âm vượt margin và tỉ lệ cặp dương dưới `pos_target`.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P04: sửa validation và margin của RS_Loss.

1. qrsiat/config: khi rs_kernel == "quantum", từ chối rs_margin >= 1.0 hoặc < 0 với
   thông báo rõ (hiện giá trị bằng 1 lọt qua). Khi rs_kernel == "cosine" giữ ràng buộc cũ.
2. models/RSIAT_adapter.py, RS_Loss: thêm rs_margin_mode ("fixed" mặc định, "quantile")
   và rs_margin_quantile (mặc định 0.9). Ở "quantile", margin dùng trong forward là EMA
   (momentum 0.9) của phân vị tương ứng của F trên cặp âm của các batch gần nhất, tính
   dưới torch.no_grad(). Thêm rs_pos_target (mặc định 1.0 để giữ hành vi cũ) thay cho
   hằng 1.0 trong relu(pos_target - F).
3. Log mỗi epoch: margin hiệu dụng, tỉ lệ cặp âm có F > margin, tỉ lệ cặp dương có
   F < pos_target.
4. Test (CPU): (a) mặc định cho kết quả giống hệt bản cũ; (b) rs_margin=1 với kernel
   quantum bị từ chối; (c) chế độ quantile trên F tổng hợp: tỉ lệ cặp âm vượt margin xấp
   xỉ 1 - quantile; (d) gradient của L_neg khác 0 khi margin < F.
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Cấu hình có `rs_margin=1` bị từ chối khi chạy `quantum`; ở chế độ `quantile`, `L_neg > 0` và có gradient. Việc có cải thiện độ chính xác hay không chỉ kết luận sau P02.

---

## P05 — Chia sẻ hoặc đóng băng kernel giữa các giai đoạn

**Ưu tiên:** cao · **Tác động:** H, K · **Phụ thuộc:** P03 · **Chi phí:** trung bình

**Vấn đề.**
- [Code + Log] Ba instance riêng `relational_kernel`, `orth_kernel`, `rs_kernel` (`RSIAT_adapter.py:83-99`). Ở base chỉ `rs_kernel` được huấn luyện và bị bỏ sau task 0 (số tham số giảm từ 61,568 xuống 49,248). `relational_kernel` và `orth_kernel` ở base chỉ nhận gradient bằng 0 qua bẫy `zero` trong `StepModule`, nên **tại task 1 teacher kernel của qrel là kernel khởi tạo ngẫu nhiên chưa huấn luyện**.
- [Code] Các kernel được huấn luyện cùng loss mà chúng đo (`W_q` nằm trong optimizer). Loss quan hệ có thể giảm bằng cách đổi kernel thay vì đổi feature.
- Kernel khác nhau ở RS, qrel, qorth nên hình học không nhất quán giữa các giai đoạn.

**Cơ sở khoa học.** Relational Knowledge Distillation (Park và cộng sự, 2019) đo quan hệ bằng một hàm khoảng cách/tương đồng cố định; nếu metric được học cùng mục tiêu thì áp lực dồn vào metric. Học kernel là một bài toán riêng (kernel-target alignment, Cristianini 2001) và Thanasilp và cộng sự (2024) nêu rằng huấn luyện embedding bằng alignment cũng gặp concentration.

**Đề xuất khắc phục.** Chính sách `kernel_policy`:
- `legacy` (hiện tại, mặc định).
- `shared_frozen`: một `PairwiseKernel` duy nhất, huấn luyện ở base bằng `L_RS` (có hiệu chỉnh P03), sau base đóng băng (`requires_grad_(False)`), dùng cho cả qrel (student và teacher cùng trọng số) và qorth.
- `fixed_calibrated`: kernel khởi tạo ngẫu nhiên + hiệu chỉnh băng thông, không bao giờ huấn luyện.
Lưu/khôi phục kernel và γ trong checkpoint.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P05: thêm kernel_policy trong {"legacy","shared_frozen","fixed_calibrated"}.

1. models/RSIAT_adapter.py (khởi tạo kernel, dòng ~71-99) và _build_step_module: với
   shared_frozen, tạo MỘT PairwiseKernel, dùng làm rs_kernel ở base. Sau khi kết thúc
   task 0 (trước khi xây StepModule của task 1) gọi requires_grad_(False) và eval(), rồi
   gán cùng đối tượng cho relational_kernel (student) và orth_kernel; relational_kernel_
   teacher là deepcopy đóng băng như hiện nay. Với fixed_calibrated, không bao giờ
   huấn luyện (requires_grad False ngay từ đầu), chỉ hiệu chỉnh băng thông (P03).
2. Kernel đóng băng không được đưa vào optimizer và không dính bẫy "zero" trong
   StepModule (bỏ khỏi vòng for parameters khi requires_grad False).
3. CheckpointManager: lưu/khôi phục trạng thái kernel (kể cả buffer bandwidth) và
   kernel_policy; resume phải từ chối khi policy khác nhau.
4. legacy phải giữ nguyên hành vi và số tham số (61,568 ở task 0; 49,248 ở task >= 1).
5. Test (CPU): (a) legacy giống hệt bản cũ; (b) shared_frozen: băm tham số kernel không
   đổi qua các task, optimizer không chứa tham số kernel từ task 1; (c) số tham số huấn
   luyện ở task >= 1 chỉ còn của aligner (24,608) + network; (d) save/resume khôi phục
   đúng kernel.
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Băm tham số kernel bất biến sau base; teacher kernel của qrel không còn là kernel ngẫu nhiên; số tham số giảm. Việc chính sách nào tốt hơn cần P02.

---

## P06 — Qorth: gradient không mong muốn và các đường thoát

**Ưu tiên:** trung bình · **Tác động:** K · **Phụ thuộc:** P05 · **Chi phí:** thấp

**Vấn đề.**
- [Code + Đo] `weights = softmax(values/τ)` không `detach` trong nhánh `orth_kernel` của `StepModule` (và trong `qrsiat/quantum/losses.py`). Sai phân hữu hạn: prototype hạng thấp (hinge = 0) có gradient âm cỡ −0.03 đến −0.06 (bị đẩy lên), prototype đầu có gradient dương ≈ 0.6–1.0. Nhỏ nhưng sai dấu.
- [Code] `aligned_prototypes = self.aligner(old_prototypes.detach(), ...)`: `detach` chỉ cắt gradient ở đầu vào; tham số aligner vẫn nhận gradient từ qorth, trong khi aligner dùng chung với `L_align`. qorth có thể dịch prototype khỏi feature mà không đổi feature.
- [Code] `orth_kernel` huấn luyện được (có thể tăng `‖W_q‖` để làm mọi F giảm).
- Lưu ý: ở thang hiện tại qorth gần như không kích hoạt (P03), nên các đường thoát chưa bộc lộ; cần sửa **trước** khi hiệu chỉnh băng thông để chúng không xuất hiện.

**Cơ sở khoa học.** Chặn gradient qua trọng số chọn lọc (như attention weights dùng làm hệ số) và qua mục tiêu tham chiếu (như target network) là thực hành chuẩn để loss chỉ tạo gradient lên đại lượng cần điều chỉnh. Nguyên tắc: mỗi loss chỉ nên đẩy gradient vào tham số mà nó được thiết kế để điều chỉnh (ở đây là adapter).

**Đề xuất khắc phục.**
1. `weights = softmax(values / τ).detach()`.
2. Tính `aligned_prototypes` dưới `torch.no_grad()` (hoặc dùng bản aligner đóng băng cập nhật theo EMA).
3. `orth_kernel` đóng băng (qua P05).
4. Hợp nhất hoặc xóa `losses.py::qorth_loss` để chỉ còn một cài đặt (P18).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P06: sửa các gradient không mong muốn của qorth, sau cờ qorth_stopgrad
(mặc định false để giữ hành vi cũ).

1. qrsiat/training/step_module.py, nhánh orth_kernel: khi cờ bật, (a) weights =
   softmax(values / temperature).detach(); (b) tính aligned_prototypes dưới
   torch.no_grad() để tham số aligner không nhận gradient từ loss_orth (aligner vẫn học
   qua loss_align). Giữ nguyên nhánh orth_kernel is None.
2. Nếu kernel_policy (P05) khác legacy thì orth_kernel không có tham số cần gradient.
3. Test (CPU): (a) cờ tắt cho cùng giá trị loss và gradient như cũ; (b) cờ bật: gradient
   của loss_orth theo tham số aligner và theo orth_kernel là None hoặc 0, theo feature
   khác 0 khi có cặp vượt epsilon; (c) kiểm tra sai phân hữu hạn: với weights detach,
   gradient theo giá trị top-k bằng weights * 1[value > epsilon].
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Test gradient đạt; không có tham số aligner/kernel nhận gradient từ `loss_orth` khi cờ bật.


---

## P07 — Tham số và phép tính vô tác dụng trong đường fidelity; baseline cổ điển trên cùng 16 góc

**Ưu tiên:** trung bình · **Tác động:** T, K · **Phụ thuộc:** không · **Chi phí:** thấp–trung bình

**Vấn đề.** [Đo, đã kiểm trên đúng logic của `simulator.py`] Fidelity `|⟨ψ_i|ψ_k⟩|²` bất biến với mọi unitary cố định áp **sau cổng mã hóa dữ liệu cuối cùng**:
- `θ` của lớp cuối có gradient đúng bằng 0 (đổi `θ_L` làm F lệch ~1e-15; đổi `θ₁` làm F lệch ~0.16).
- Chuỗi CNOT cuối (kể cả CNOT vòng khi `cycle=True`) không làm đổi F (lệch ~1e-15).
- Với L=1 mạch bằng công thức đóng `∏ cos²(Δα/2)`, không cần mô phỏng 2^q chiều.
- Với L=2, 8 trên 16 giá trị `θ` vô dụng; phần "mạch học được" chỉ còn `θ₁` (8 số) cộng `W_q, b_q`.
- [Code] `QuantumFeatureMap.forward` luôn tính `readout` rồi `PairwiseKernel` bỏ đi (`states, _ = ...`).
Riêng aligner dùng readout Pauli (không bất biến theo unitary) nên `θ_L` và CNOT cuối **còn tác dụng** ở đó: không được bỏ.

**Cơ sở khoa học.** |⟨Uψ|Uφ⟩| = |⟨ψ|φ⟩| với U unitary (tính chất hiển nhiên của unitary). Havlíček và cộng sự (2019) và Schuld, Sweke, Meyer (2021) về fidelity kernel và vai trò của mã hóa dữ liệu. Bowles và cộng sự (2024): bỏ rối thường không giảm hiệu năng. Huang và cộng sự (2021): fidelity là phép đo toàn cục, dễ làm K ≈ I; **projected kernel** (dựa trên kỳ vọng cục bộ) là lựa chọn đã được nghiên cứu. Thanasilp và cộng sự (2024): phép đo toàn cục là một nguồn concentration.

**Đề xuất khắc phục.**
1. `QuantumFeatureMap`: tham số `final_entangler=False` và `return_readout=False` cho đường kernel; đặt mask cho `θ[-1]` (không đưa vào optimizer).
2. Tùy chọn `trainable_mixer`: khối `V_φ` (RY + CNOT, không phụ thuộc dữ liệu) **giữa** lớp mã hóa 1 và 2, thay vai trò của `θ`; chỉ khối đặt giữa các lớp mã hóa mới có tác dụng.
3. Thêm hai kernel đối chứng: `projected_rbf` (RBF trên 16 góc sau `tanh`, không có mạch) và `projected_pauli` (RBF trên readout X/Z, kiểu projected quantum kernel). Dùng để tách đóng góp của mạch khỏi đóng góp của phép chiếu (P02, cấu hình C3).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P07: loại bỏ phần vô tác dụng khỏi đường fidelity và thêm kernel đối chứng.

1. qrsiat/quantum/feature_map.py và simulator.py: thêm tham số final_entangler (mặc định
   True) và return_readout (mặc định True). PairwiseKernel (kernels.py) dùng
   final_entangler=False và return_readout=False. Aligner (aligner.py) giữ nguyên, vì
   readout Pauli phụ thuộc CNOT cuối và theta lớp cuối. Trước khi bỏ, grep xem có nơi nào
   khác dùng readout của QuantumFeatureMap không.
2. QuantumFeatureMap: khi đường kernel bật cờ drop_dead_theta, không đăng ký theta[-1]
   như tham số huấn luyện (ví dụ dùng mask hoặc tách thành parameter và buffer). Giữ
   checkpoint cũ nạp được (strict=False có log).
3. Thêm kernel mới trong PairwiseKernel: "projected_rbf" (project -> tanh*pi -> RBF trên
   các góc, băng thông là buffer) và "projected_pauli" (RBF trên readout X/Z của state).
   Cùng giao diện [B,B].
4. Tùy chọn trainable_mixer (mặc định false): thêm khối V_phi giữa lớp 1 và lớp 2 gồm
   tầng RY với góc huấn luyện được và chuỗi CNOT.
5. Test (CPU): (a) F bất biến khi đổi theta[-1] và khi bỏ CNOT cuối (atol 1e-6), bao gồm
   cycle=True; (b) readout của aligner KHÔNG bất biến khi bỏ CNOT cuối (test khẳng định
   đúng); (c) L=1 trùng công thức đóng prod(cos^2(delta/2)); (d) projected_* trả về
   [B,B] trong [0,1], đường chéo bằng 1; (e) gradient theo theta[-1] bằng 0 ở đường
   kernel (xác nhận bằng autograd), khác 0 ở theta[0].
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Test (a)–(e) đạt; số tham số huấn luyện của kernel giảm đúng 8 mỗi kernel (L=2); thời gian mô phỏng giảm (đo bằng thời gian tường, P17).

---

## P08 — Centering nhất quán giữa các đường

**Ưu tiên:** trung bình · **Tác động:** H · **Phụ thuộc:** P03 · **Chi phí:** trung bình

**Vấn đề.** [Code] Tâm là trung bình batch, **không detach**, và mỗi đường dùng một tâm khác:
- RS: tâm batch của feature đã normalize.
- qrel: student dùng tâm batch student (đã gather), teacher dùng tâm batch teacher (tâm riêng, khác nhau).
- qorth (cross): trung bình của tâm feature batch và tâm prototype (`kernels.py`), phụ thuộc số lớp cũ; có gradient chảy về `aligned_prototypes`.
- aligner: tâm batch teacher (đúng với prototype, nhưng phụ thuộc batch).
- DDP: qorth dùng feature cục bộ, qrel dùng feature đã gather (`pair_gather` mặc định tắt).
- Batch có một mẫu thì không center (`values.shape[0] > 1`).
Hệ quả: cùng một cặp ảnh cho F khác nhau tùy thành phần batch; thang đo giữa train và so sánh không ổn định.

**Cơ sở khoa học.** Căn chỉnh kernel có tâm (Cortes và cộng sự, 2012) định nghĩa tâm trên toàn bộ phân phối, không phải trên một batch. Batch Normalization (Ioffe và Szegedy, 2015) dùng thống kê trượt (running statistics) để loại phụ thuộc batch.

**Đề xuất khắc phục.** `center_mode ∈ {batch, ema, fixed}`:
- `ema`: buffer cập nhật `torch.no_grad()` với momentum 0.99, lưu trong checkpoint.
- `fixed`: trung bình feature của task hiện tại, tính một lần trước task (ảnh không augmentation; có thể tái dùng lượt trích feature đã có cho drift).
- Mọi tâm được `detach`. qorth dùng cùng một tâm cho feature và prototype (không trộn trung bình prototype). DDP: tâm EMA/fixed không cần gather.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P08: thêm center_mode trong {"batch","ema","fixed"} (mặc định "batch").

1. qrsiat/quantum/feature_map.py, aligner.py, kernels.py: thêm cơ chế tâm. "batch": giữ
   hành vi cũ nhưng .detach() tâm khi cờ center_detach bật (mặc định false để giữ bit-
   wise). "ema": buffer center với cập nhật EMA (momentum 0.99) bằng no_grad ở chế độ
   train, dùng buffer ở eval. "fixed": nhận center từ bên ngoài (set_center).
2. kernels.py cross-branch (qorth): với ema/fixed dùng CÙNG một tâm cho features và
   prototypes thay vì trung bình hai tâm.
3. models/RSIAT_adapter.py: với "fixed", tính tâm từ feature task hiện tại (không
   augmentation) ở đầu task (tái dùng lượt trích feature hiện có nếu có thể), truyền cho
   mọi kernel và aligner của task.
4. CheckpointManager lưu/khôi phục buffer tâm.
5. Test (CPU): (a) batch + không detach giống hệt bản cũ; (b) ema/fixed: F của cùng một
   cặp mẫu không phụ thuộc mẫu còn lại trong batch (hoán đổi các mẫu khác, F không đổi),
   còn ở "batch" thì có đổi; (c) gradient không chảy qua tâm khi detach.
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Độ lệch chuẩn của F cho một cặp cố định giữa các batch khác nhau giảm gần 0 ở `ema`/`fixed` (đo bằng P01).

---

## P09 — Optimizer: param_groups, nhóm lượng tử, gradient clipping

**Ưu tiên:** trung bình · **Tác động:** H, K · **Phụ thuộc:** P05 · **Chi phí:** thấp

**Vấn đề.**
- [Code] Cả hai nhánh `adam` (`RSIAT_adapter.py:552-557` và `:581-586`) bỏ qua toàn bộ `param_groups`, dùng `AdamW(self._trainable_step_parameters(), lr=init_lr, weight_decay)`. Hệ quả: mất lr cứng 0.01 của base, `ae_init_lr` và `ae_weight_decay` của baseline `rae`. Kết quả `adam` và `sgd` khác nhau cả về cấu trúc LR, không chỉ thuật toán, và baseline `rae` bị ảnh hưởng nhiều nhất.
- [Code] `_append_step_parameters` (`:599-620`) đưa mọi tham số lượng tử vào một nhóm có cùng `lr` và `weight_decay` với network (`:617-618`); không có weight decay riêng cho `θ`, `b_q`, `W_up`.
- [Code] Không có gradient clipping (có GradScaler fp16).
- [Suy luận] Lần chạy hiện dùng SGD lr 0.01; ở vùng phẳng của RS (P03) gradient của `W_q` rất nhỏ nên SGD gần như không làm `W_q` di chuyển trong 120–540 bước, còn Adam (bước ∝ LR không phụ thuộc độ lớn gradient) thoát nhanh hơn: lựa chọn optimizer quyết định nhánh có "sống" không.

**Cơ sở khoa học.** AdamW tách weight decay khỏi cập nhật thích nghi (Loshchilov và Hutter, 2019). Tham số góc/tuần hoàn và tham số khởi tạo bằng 0 (`W_up`) không nên bị weight decay kéo về 0. Cắt gradient theo norm là biện pháp chuẩn khi quy mô gradient thay đổi theo hệ số warm-up của loss.

**Đề xuất khắc phục.**
1. Nhánh `adam` dùng `AdamW(param_groups, ...)` để tôn trọng LR từng nhóm.
2. Nhóm `quantum` riêng với `quantum_lr` và `quantum_weight_decay` (mặc định 0), tách `W_q`, `W_down` (có thể decay nhẹ) khỏi `θ`, `b_q`, `W_up` (decay 0).
3. `grad_clip_norm` (mặc định `None`): gọi sau `scaler.unscale_(optimizer)` và trước `scaler.step`.
4. Log norm gradient theo nhóm.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P09: sửa tạo optimizer trong models/RSIAT_adapter.py (_train, _append_step_
parameters) mà không đổi hành vi mặc định của nhánh sgd.

1. Nhánh adam (task 0 và task >= 1): dùng optim.AdamW(param_groups, lr=..., weight_decay=
   ...) thay cho AdamW(self._trainable_step_parameters(), ...). Giữ cờ legacy_adam_groups
   (mặc định false) nếu cần tái hiện kết quả adam cũ.
2. _append_step_parameters: tách tham số lượng tử khỏi nhóm "extras" thành nhóm riêng với
   quantum_lr (mặc định self.init_lr * lr_scale, để giữ hành vi cũ) và
   quantum_weight_decay (mặc định self.weight_decay để giữ hành vi cũ). Nhóm con cho
   theta, bias của project/down và up.weight với weight_decay = 0 khi cờ
   quantum_no_decay bật.
3. Thêm grad_clip_norm (mặc định None): trong _init_train, sau scaler.unscale_(optimizer)
   (nếu có scaler) và trước step, clip_grad_norm_ trên tham số có gradient; log norm.
4. In log số tham số, lr và weight_decay thực tế của từng nhóm (log_count_parameter hiện
   chỉ đếm tham số).
5. Test (CPU): (a) mặc định sgd: danh sách nhóm, lr, wd giống hệt bản cũ; (b) adam mới:
   lr của từng nhóm đúng theo cấu hình (không còn một nhóm duy nhất); (c) clipping chỉ
   chạy khi cờ bật và đúng thứ tự với GradScaler (unscale trước clip).
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Test đạt; log hiển thị lr/wd từng nhóm; kết quả `sgd` mặc định không đổi.

---

## P10 — Lịch warm-up và trọng số loss nhất quán

**Ưu tiên:** trung bình · **Tác động:** H, K · **Phụ thuộc:** P02 · **Chi phí:** thấp

**Vấn đề.**
- [Code] Base task tính warm-up inline `min(1, epoch/max(1, W))` (`RSIAT_adapter.py:730-732`), trong khi `qhybrid` dùng `warmup_value` (`:735-736`); hai cách khác nhau khi `W = 0`. `task_loss_weights` trong `schedule.py` không được dùng.
- [Code] `rae` dùng hằng số `beta`, `gamma` (`:739-740`), còn `qhybrid` warm-up từ 0: so sánh hai chế độ bị nhiễu bởi lịch loss, không chỉ bởi phương pháp.
- [Code + Log] Epoch 0 của mỗi task incremental: `beta = gamma = 0` nên không có regularizer nào (thậm chí không gọi network cũ); với `W = 3`, `inc_epochs = 20`, trọng số trung bình bằng 0.9 giá trị đầy đủ. Adapter có thể trôi trước khi ràng buộc chống quên có hiệu lực.

**Cơ sở khoa học.** Warm-up tốc độ học (Goyal và cộng sự, 2017) xử lý bất ổn khi gradient lớn lúc đầu. Không có lý do tương tự cho regularizer chống quên: nó cần có hiệu lực ngay khi adapter bắt đầu thay đổi. Aligner đã có khởi động mềm sẵn nhờ `W_up = 0` (residual bằng 0). Đây là bài toán ổn định–dẻo: warm-up hợp lý cho thành phần tạo dẻo (như `lambda_rs`), đáng nghi cho thành phần bảo vệ ổn định (`beta`, `gamma`). **Đây là giả thuyết cần ablation**, không phải khẳng định.

**Đề xuất khắc phục.**
1. Một hàm lịch duy nhất `loss_schedule(name, epoch, ...)` trong `schedule.py`, dùng ở mọi nơi (kể cả base).
2. `warmup_mode` riêng cho từng hệ số: `lambda_rs` mặc định `linear`; `beta`, `gamma` có `linear` (hiện tại), `none`, `start_fraction` (bắt đầu từ một phần giá trị, ví dụ 0.5).
3. `rae` và `qhybrid` dùng cùng lịch khi so sánh (cờ `schedule_parity`).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P10: hợp nhất lịch trọng số loss.

1. qrsiat/training/schedule.py: thêm loss_schedule(base, epoch, warmup_epochs, mode,
   start_fraction=0.0) với mode "linear" (hành vi cũ), "none", "start_fraction". Xử lý
   warmup_epochs == 0 giống nhau ở mọi nơi (trả về base). Dọn task_loss_weights: dùng nó
   hoặc xóa nó.
2. models/RSIAT_adapter.py _compute_rt_loss: thay phép tính inline ở nhánh base và nhánh
   rae/qhybrid bằng loss_schedule. Mặc định: lambda_rs "linear", beta/gamma của qhybrid
   "linear", rae giữ hằng số (đúng hành vi cũ). Thêm cờ schedule_parity: khi bật, rae dùng
   cùng warmup_mode của qhybrid.
3. Thêm cờ beta_warmup_mode, gamma_warmup_mode, lambda_rs_warmup_mode,
   beta_start_fraction, gamma_start_fraction, validate sớm trong qrsiat/config.
4. Test (CPU): giá trị ở epoch 0..W+1 cho từng mode; W=0; hành vi mặc định trùng bản cũ
   ở mọi epoch; không còn công thức warm-up inline nào ngoài schedule.py.
Báo cáo theo mẫu P00, nhắc người dùng rằng việc warmup none/start_fraction có tốt hơn hay
không chỉ kết luận được sau ablation nhiều seed (P02).
```

**Nghiệm thu.** Chỉ còn một nguồn công thức warm-up; mặc định không đổi; có cờ để chạy ablation `none` so với `linear` ở `beta`, `gamma`.

---

## P11 — Chuẩn hóa `L_align` và khả năng học của aligner

**Ưu tiên:** trung bình · **Tác động:** H, K · **Phụ thuộc:** P01 · **Chi phí:** thấp

**Vấn đề.**
- [Code] `loss_align = mse_loss(features, projected_old)` là trung bình trên `B·768` phần tử, trong khi `representation.md` ghi `(1/B)Σ‖·‖²`: hệ số `beta` hiệu dụng nhỏ hơn 768 lần so với công thức; tài liệu và mã không khớp.
- [Code] Gradient tới `W_up` tỉ lệ `(2/d)·E[e·r]` (rất nhỏ); `W_up = 0` nên gradient tới `down` và `θ` bằng 0 cho đến khi `W_up ≠ 0`. Với SGD lr 0.01 và chỉ 120–540 bước, aligner có thể gần như vẫn là hàm đồng nhất. Hiện không có số đo nào cho biết aligner học được gì.
- [Code] Phần hiệu chỉnh `W_up · r` nằm trong không gian con tối đa 16 chiều (`up` là 768×16), trong khi drift của feature 768 chiều không như vậy.

**Cơ sở khoa học.** Khởi tạo residual bằng 0 (Goyal và cộng sự, 2017; LoRA, Hu và cộng sự, 2022) hợp lý nhưng cần LR và optimizer đủ để thoát khỏi điểm khởi tạo. Thang của loss không nên phụ thuộc số chiều d, hoặc phải ghi rõ.

**Đề xuất khắc phục.**
1. Tham số `align_reduction ∈ {mean, per_sample}`: `mean` giữ hành vi cũ; `per_sample` = `Σ_d` rồi mean theo mẫu (khớp tài liệu), kèm ghi chú: để tương đương số với `mean` cần `beta_new = beta / 768`.
2. `aligner_lr` riêng (có thể lớn hơn network) và tùy chọn khởi tạo `up` nhỏ (std 1e-3) để gradient tới `down` không bằng 0.
3. Chỉ số bắt buộc: `R² = 1 − ‖f_t − P(f_{t−1})‖² / ‖f_t − f_{t−1}‖²` (dương nghĩa là aligner đang tốt hơn hàm đồng nhất), log cùng `‖W_up‖`.
4. Nếu `R² ≈ 0` ổn định: aligner không có giá trị và nên thay bằng cách cổ điển hoặc bỏ.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P11: làm rõ thang của L_align và cho aligner khả năng học đo được.

1. qrsiat/training/step_module.py: thêm align_reduction ("mean" mặc định = hành vi cũ,
   "per_sample" = tổng theo chiều rồi trung bình theo mẫu). Ghi docstring: beta tương
   đương = beta/768 giữa hai chế độ.
2. qrsiat/quantum/aligner.py: thêm tùy chọn up_init_std (mặc định 0.0 = hành vi cũ).
3. models/RSIAT_adapter.py: thêm aligner_lr riêng trong param_groups (mặc định bằng LR
   hiện tại của nhóm extras để giữ hành vi cũ).
4. Tính và log mỗi epoch: R2_align = 1 - ||f_t - P(f_{t-1})||^2 / ||f_t - f_{t-1}||^2
   (trên batch hiện tại, no_grad) và ||W_up||_F. Gắn vào diagnostics của P01.
5. Test (CPU): (a) mặc định giống hệt bản cũ; (b) per_sample bằng mean * 768 trên dữ liệu
   ngẫu nhiên; (c) R2 = 0 khi aligner là hàm đồng nhất, R2 > 0 khi aligner đã khớp phần
   dịch chuyển trên dữ liệu tổng hợp.
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Có `R2_align` theo task. Quyết định giữ/bỏ aligner dựa trên số này kết hợp P02.

---

## P12 — `sigma = 4.0` cố định trong SSCA và nguy cơ trọng số suy biến

**Ưu tiên:** **cao** · **Tác động:** H · **Phụ thuộc:** P01 · **Chi phí:** thấp

**Vấn đề.**
- [Code] `estimate_drift(..., sigma=4.0)` (`RSIAT_adapter.py:288-294`): `σ` cứng. Trọng số `w = exp(−‖f_old − μ_c‖²/(2σ²))`.
- [Log] `radius = 1.574` nghĩa là `‖f − μ_lớp‖ ≈ 43.6`, tức khoảng cách bình phương cỡ 1900 trở lên (giữa các lớp còn lớn hơn). Với `σ² = 16`, `w ≈ exp(−1900/32) ≈ 1e-26` hoặc nhỏ hơn.
- [Code] Hàm cũ `displacement` trong `base.py:386-394` cộng thêm `+1e-5` vào trọng số: nếu `estimate_drift` làm tương tự, trọng số bị nền `1e-5` chi phối và **drift suy biến thành trung bình toàn cục** (mọi lớp cũ nhận cùng một độ dịch). Nếu không có nền, trọng số chuẩn hóa sẽ tập trung vào 1–2 mẫu gần nhất (ước lượng nhiễu). Nếu tính bằng float32, `exp` underflow về 0.
- [Suy luận] Chưa đọc `qrsiat/stats/drift.py`, nên chưa biết trường hợp nào xảy ra. **Phải đo** kích thước mẫu hiệu dụng (ESS) trước khi kết luận.

**Cơ sở khoa học.** Semantic Drift Compensation (Yu và cộng sự, CVPR 2020) ước lượng drift của prototype bằng trung bình có trọng số của vector drift của dữ liệu task hiện tại, với trọng số theo khoảng cách tới prototype; băng thông của nhân Gaussian là siêu tham số cần khớp với thang khoảng cách. Heuristic trung vị (Garreau và cộng sự, 2017) là cách đặt băng thông theo thang dữ liệu. Phương pháp đánh giá chuẩn: so ước lượng drift với drift thật (prototype tính lại bằng mạng mới trên dữ liệu cũ, chỉ dùng chẩn đoán offline; `_compute_class_mean` đã có tham số `oracle`).

**Đề xuất khắc phục.**
1. Log ESS theo lớp `ESS_c = 1 / Σ_i w_ic²` (trọng số đã chuẩn hóa theo mẫu) và tỉ lệ trọng số nằm trên nền `1e-5`.
2. Tính trọng số trong không gian log bằng softmax theo mẫu (ổn định số, không underflow, không cần nền). Xem Phụ lục A7.
3. `sigma_mode ∈ {fixed, median}`: `median` đặt `σ² = κ·median(d²)` với `κ ∈ {0.25, 0.5, 1}`; chọn `κ` sao cho ESS ≥ 10–25% số mẫu.
4. Chẩn đoán offline: sai số tương đối `‖μ̂ − μ*‖ / ‖μ* − μ_cũ‖` của ssca, so với không bù, trên một vài task (cần dữ liệu cũ, chỉ dùng chẩn đoán).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P12: chẩn đoán và sửa trọng số drift (ssca). Trước hết ĐỌC qrsiat/stats/drift.py
và báo cáo công thức thực tế (có nền 1e-5 không, dtype, cách chuẩn hóa, cách all-reduce).

1. qrsiat/stats/drift.py: thêm hàm estimate_drift_stable (tham khảo Phụ lục A7) tính trọng
   số bằng softmax theo mẫu trên -d2/(2*sigma2) (float64), trả về (delta, ess, sigma2).
   Hỗ trợ DDP bằng cách all-reduce max rồi tổng số mũ ổn định (logsumexp), không cộng nền
   1e-5. Giữ estimate_drift cũ làm mặc định.
2. models/RSIAT_adapter.py (dòng ~288-297): thêm cờ drift_sigma_mode ("fixed" mặc định,
   sigma=4.0 như cũ; "median") và drift_sigma_kappa (mặc định 0.5). Log: sigma hiệu dụng,
   phân vị 5/50/95 của ESS theo lớp, ESS/N, và tỉ lệ trọng số bị chi phối bởi nền (nếu hàm
   cũ có nền).
3. Cờ diagnostics_drift_oracle (mặc định false): ở chế độ chẩn đoán offline, tính
   prototype "thật" bằng mạng mới trên dữ liệu lớp cũ và log sai số tương đối của ssca so
   với không bù. Chỉ dùng để chẩn đoán, không được dùng khi huấn luyện thật.
4. Test (CPU): (a) trên dữ liệu tổng hợp với khoảng cách >> sigma, bản ổn định không NaN
   và không underflow; (b) khi sigma lớn, drift tiến tới trung bình toàn cục; khi sigma nhỏ
   tiến tới láng giềng gần nhất; (c) ESS nằm trong [1, N]; (d) mặc định không đổi kết quả.
Báo cáo theo mẫu P00, nêu rõ kết luận về estimate_drift cũ (có suy biến không).
```

**Nghiệm thu.** Có ESS theo lớp trong log. Nếu ESS ≈ 1 hoặc ≈ N (nền chi phối), đặt `drift_sigma_mode=median` và so sánh độ chính xác lớp cũ (P02). **Đây có thể là cải tiến độ chính xác độc lập với phần lượng tử.**


---

## P13 — (Thực nghiệm) Dùng aligner để cập nhật prototype, so với SSCA

**Ưu tiên:** thấp (chỉ sau P11, P12) · **Tác động:** H · **Phụ thuộc:** P11, P12 · **Chi phí:** trung bình

**Vấn đề.** [Code] Aligner chỉ tham gia `L_align` và qorth. Prototype lớp cũ được cập nhật bằng SSCA cổ điển (`RSIAT_adapter.py:295-297`), còn suy luận không dùng aligner. Vì vậy phần "lượng tử" của aligner chỉ ảnh hưởng độ chính xác qua gradient lên adapter, và `representation.md` mô tả ánh xạ prototype qua aligner nhưng ánh xạ này không bao giờ ghi lại vào `_class_means`.

**Cơ sở khoa học.** SDC (Yu và cộng sự, 2020) ước lượng drift bằng trung bình có trọng số cục bộ. Aligner `P(f) = f + W_up·r(f)` học ánh xạ teacher → student theo cách có tham số, nên có thể coi là một ước lượng drift học được (bị giới hạn ở hạng ≤ 16 và `|r| ≤ 1`). Câu hỏi khoa học là liệu nó ước lượng drift tốt hơn SDC hay không; phép đo chuẩn là sai số so với prototype thật (chẩn đoán offline, P12).

**Đề xuất khắc phục.** Thử ba cách cập nhật `μ_c`: (i) SSCA (hiện tại), (ii) `μ_c ← P(μ_c)` với tâm giống lúc huấn luyện aligner, (iii) trộn có trọng số `α·SSCA + (1−α)·P`. Đánh giá bằng sai số tương đối so với prototype thật và bằng độ chính xác lớp cũ.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P13: thêm cờ prototype_update trong {"ssca","aligner","mix"} (mặc định "ssca").

1. models/RSIAT_adapter.py (sau huấn luyện mỗi task, quanh dòng 286-297): với "aligner",
   tính aligned = aligner(prototypes_cu, centering_mean=teacher_feature_mean_cua_task)
   dưới no_grad và full precision, rồi gán _class_means[:known] = aligned. Với "mix", dùng
   alpha * (mean + gap_ssca) + (1 - alpha) * aligned (cờ prototype_mix_alpha, mặc định
   0.5). Tâm phải giống tâm dùng khi huấn luyện aligner (xem P08).
2. Log sai số tương đối so với prototype thật khi diagnostics_drift_oracle bật (P12), cho
   cả ba cách tính trên cùng một task (chỉ tính, chỉ ssca được áp dụng khi oracle bật).
3. Test (CPU): "ssca" giống hệt bản cũ; "aligner" với aligner đồng nhất (W_up = 0) cho
   prototype không đổi; "mix" với alpha=1 bằng "ssca".
Báo cáo theo mẫu P00. Nhắc người dùng đây là thực nghiệm: không đặt làm mặc định nếu
chưa có kết quả nhiều seed.
```

**Nghiệm thu.** Có bảng sai số prototype của ba cách; chỉ giữ cách nào tốt hơn ssca có ý nghĩa thống kê.

---

## P14 — Covariance lớp cũ không cập nhật; covariance hạng thấp ở lớp ít dữ liệu

**Ưu tiên:** cao · **Tác động:** H · **Phụ thuộc:** P12 · **Chi phí:** trung bình

**Vấn đề.**
- [Code] Sau mỗi task chỉ `_class_means[:known] += gap` (`RSIAT_adapter.py:296-297`). `_class_covs[:known]` **không bao giờ được cập nhật theo drift**: `_compute_class_mean` chỉ ghi lát của lớp mới (`base.py:349-351`); hàm `displacement_cov` (`base.py:361-384`) không được gọi và gọi `self.cov_computation` chưa thấy định nghĩa. CA lấy mẫu quanh mean đã dịch nhưng với covariance cũ.
- [Log + Code] Lớp có 13–20 ảnh cho covariance hạng ≤ 12–19 cộng `ε·I` (`covariance_epsilon=1e-3` và sàn eigenvalue ở `base.py:50-75`). Mẫu Gaussian chỉ biến thiên trong không gian con nhỏ, phân bố tổng hợp quá hẹp so với phân bố thật. Phù hợp (chưa chứng minh) với việc lớp mới học từ ít dữ liệu rất yếu (task 5: 55.9%, task 6: 42.7%).
- [Code] `_class_covs` lưu 768×768 float32 mỗi lớp: ≈ 472 MB ở 200 lớp, ghi vào mọi checkpoint.

**Cơ sở khoa học.** FeCAM (Goswami và cộng sự, NeurIPS 2023) cho thấy phân bố đặc trưng của các lớp không đồng nhất, mô hình hóa covariance tốt hơn lấy mẫu Gaussian + classifier tuyến tính, và dùng co (shrinkage) covariance và chuẩn hóa tương quan để có ma trận đủ hạng ổn định. Phân tích phân biệt có điều chuẩn (Friedman, 1989) co covariance từng lớp về covariance gộp; ước lượng co của Ledoit và Wolf (2004) là chuẩn cho `n < d`. Với hạng thấp, trọng số co nên tăng khi `n_c` giảm.

**Đề xuất khắc phục.**
1. Co về covariance gộp: `Σ_c' = ((n_c−1)·Σ_c + ν·Σ_pool) / (n_c−1+ν)`, với `ν` chọn bằng validation (ví dụ `{0, 10, 30, 100}`). Xem Phụ lục A6. Cần lưu `n_c`.
2. Chỉ gộp trong cùng hệ tọa độ feature: ưu tiên `Σ_pool` của các lớp **cùng task** (tránh trộn hệ tọa độ cũ/mới); so sánh với gộp toàn bộ.
3. Với covariance lớp cũ: thử (a) giữ nguyên, (b) co về `Σ_pool` của task hiện tại, (c) co kèm mean đã dịch. Đo bằng sai số Frobenius so với covariance thật (chẩn đoán offline).
4. Lưu trữ gọn: thừa số hạng thấp `(U, λ, ε)` cho từng lớp (hạng ≤ `n_c − 1` ≤ 62 so với 768) cộng một ma trận gộp đang chạy; hoặc lưu nửa tam giác (FeCAM làm vậy).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P14: co covariance cho lớp ít dữ liệu và thử cập nhật covariance lớp cũ.

1. qrsiat/stats/accumulators.py và models/base.py: lưu số mẫu n_c của từng lớp cùng
   covariance (vào metadata checkpoint). Thêm hàm shrink_covariances (Phụ lục A6) và cờ
   cov_shrink_nu (mặc định 0.0 = hành vi cũ), cov_shrink_pool ("task" hoặc "all").
2. models/base.py: trong _stage2_compact_classifier_local, trước khi lấy mẫu dùng
   covariance đã co (cache, xem P15). Không ghi đè _class_covs gốc.
3. Chẩn đoán offline (cờ diagnostics_cov_oracle, mặc định false): với dữ liệu lớp cũ, tính
   covariance thật bằng mạng mới và log sai số Frobenius tương đối của (a) giữ nguyên,
   (b) co về Sigma_pool task hiện tại. Chỉ chẩn đoán, không dùng khi huấn luyện thật.
4. Lưu trữ: tùy chọn cov_storage ("full" mặc định, "lowrank"): lưu (U, lambda, eps) theo
   từng lớp bằng eigh/SVD, tái tạo khi cần. Kiểm tra sai số tái tạo < 1e-5 tương đối.
5. Test (CPU): (a) nu=0 giống hệt bản cũ; (b) với n_c < d, sau co ma trận đủ hạng và
   xác định dương, mẫu Gaussian đủ chiều; (c) hạng của covariance co tăng khi nu tăng;
   (d) lowrank tái tạo đúng.
Báo cáo theo mẫu P00. Nhắc người dùng chọn nu trên tập validation tách từ dữ liệu train,
KHÔNG chọn bằng độ chính xác trên tập test.
```

**Nghiệm thu.** Độ chính xác lớp mới ở các task ít dữ liệu cải thiện có ý nghĩa thống kê (P02) mà độ chính xác lớp cũ không giảm; kích thước checkpoint giảm khi dùng `lowrank`.

---

## P15 — Căn chỉnh classifier (CA): chi phí lặp lại, đánh giá mỗi epoch, ánh xạ task

**Ưu tiên:** **cao** · **Tác động:** T, H · **Phụ thuộc:** không (kết hợp P14) · **Chi phí:** thấp–trung bình

**Vấn đề.**
- [Code] Trong mỗi epoch CA, vòng `for c_id in range(crct_num)` gọi `_make_sampling_covariance` (một `eigh` 768×768 float64) rồi tạo `MultivariateNormal` (thêm một Cholesky) cho **mọi lớp, mọi epoch** (`base.py:117-128`), dù covariance không đổi trong CA. Với 200 lớp × 10 epoch = 2000 lần mỗi task.
- [Code] `_compute_accuracy` chạy trên toàn bộ tập test **sau mỗi epoch CA** (`base.py:170`), fp32.
- [Log] Giai đoạn sau huấn luyện tăng gần tuyến tính theo số lớp (151 → 363 giây). Độ chính xác trong CA thường bão hòa sau 1–5 epoch (task 4: 76.87 → 77.34 sau 5 epoch rồi phẳng; task 9: 66.67 → 66.93 rồi dao động), trong khi `Loss` vẫn giảm.
- [Code] `t_id = c_id // task_size` (`base.py:118`) chỉ đúng khi mọi task cùng kích thước (đúng với 20/20, sai với `init_cls ≠ increment`).
- [Code] Mean của lớp cũ bị nhân `0.9 + decay` (`base.py:119-120`), thu nhỏ mean lớp cũ tới 10% về gốc: heuristic kế thừa, chưa có ablation.

**Cơ sở khoa học.** Chuẩn tắc số học: phân rã ma trận cố định chỉ cần tính một lần; mẫu từ `N(μ, Σ)` bằng `μ + L·z` với `L` là thừa số Cholesky. Chọn số epoch/siêu tham số bằng tập validation, không bằng tập test (tránh rò rỉ). SLCA (Zhang và cộng sự, 2023) là nguồn gốc của căn chỉnh classifier bằng mẫu Gaussian; đây là heuristic có siêu tham số cần kiểm tra.

**Đề xuất khắc phục.**
1. Cache thừa số Cholesky theo lớp, tính một lần mỗi task và dùng lại qua các task cho lớp có covariance không đổi (Phụ lục A4); `MultivariateNormal(mean, scale_tril=L)`.
2. Bỏ `_compute_accuracy` khỏi vòng CA (chỉ đánh giá ở cuối, hoặc mỗi `eval_every`).
3. Đặt `ca_epochs` nhỏ (ví dụ 3–5) **chọn trên validation tách từ dữ liệu train**, không chọn trên test.
4. Ánh xạ `lớp → task` từ danh sách kích thước task thật (`data_manager.get_task_size(t)`), không giả định đều.
5. Ablation `ca_mean_scale ∈ {legacy, none}` cho hệ số `0.9 + decay`.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P15: tối ưu và sửa logic CA trong models/base.py
(_stage2_compact_classifier_local) mà không đổi hành vi mặc định về mặt thống kê.

1. Thêm cache thừa số Cholesky theo lớp (self._scale_tril_cache; xem Phụ lục A4). Tính
   một lần mỗi task cho lớp mới, tái dùng cho lớp cũ trong các task sau; vô hiệu hóa mục
   cache khi covariance của lớp thay đổi (P14). Dùng
   MultivariateNormal(mean, scale_tril=L). Nếu Cholesky thất bại, thêm jitter nhỏ và log.
2. Bỏ self._compute_accuracy khỏi vòng epoch CA. Thêm cờ ca_eval_every (mặc định 0 =
   chỉ đánh giá sau CA; đặt 1 để giữ log cũ mỗi epoch).
3. Thêm cờ ca_class_to_task: thay t_id = c_id // task_size bằng ánh xạ xây từ kích thước
   task thực của data_manager. Mặc định "legacy" giữ công thức cũ (giống nhau khi các task
   đều bằng nhau).
4. Thêm cờ ca_mean_scale ("legacy" mặc định = 0.9 + decay; "none" = 1.0).
5. Test (CPU): (a) với seed cố định, mẫu từ scale_tril cache cho cùng phân bố (kiểm tra
   mean và covariance mẫu lớn khớp, sai số thống kê nhỏ) so với đường cũ; (b) với các task
   không đều (ví dụ 50 + 10 x 5) ánh xạ mới đúng, "legacy" sai có chủ đích; (c) số lần
   eigh mỗi task giảm từ C*ca_epochs xuống không quá C (đếm bằng mock).
6. Đo và in thời gian từng pha (lấy mẫu, huấn luyện fc, đánh giá) mỗi epoch CA.
Báo cáo theo mẫu P00; đề xuất ca_epochs chọn qua validation, không qua test.
```

**Nghiệm thu.** Thời gian CA mỗi task giảm rõ (đo bằng log pha); độ chính xác không đổi ngoài nhiễu seed (P02). Ước tính tiết kiệm: một phần đáng kể của ~40 phút sau huấn luyện (cần đo).

---

## P16 — Chi phí đánh giá, trích thống kê lớp và đồng bộ mỗi bước

**Ưu tiên:** **cao nhất về thời gian** · **Tác động:** T · **Phụ thuộc:** không · **Chi phí:** thấp

**Vấn đề.**
- [Code + Log] `_init_train` gọi `_compute_accuracy` trên toàn bộ tập test **mỗi epoch** (`RSIAT_adapter.py:700`); hàm này chạy fp32, không autocast (`base.py:218-232`) dù huấn luyện dùng AMP fp16. Thời gian mỗi epoch không phụ thuộc số bước (task 5: 6 bước/epoch vẫn 20.8 giây), cho thấy phần cố định ≥ ~18 giây/epoch. Cộng 20 epoch × 10 task ≈ 60 phút tiềm năng.
- [Code] `_compute_class_mean` tạo một `DataLoader` mới cho **từng lớp mới** (`base.py:305-333`, 20 lần mỗi task), mỗi lần khởi động worker; trong khi `accumulator.update` đã nhận nhãn nên một lượt duyệt là đủ.
- [Code] `loss.item()`, `loss_c.item()`, `loss_rt.item()` và `.cpu()` mỗi bước (`:687-691`) gây đồng bộ GPU (P01 xử lý phần log).
- [Code] `eval_batch_size = 48` trong `RuntimePlan` (log) dù không có gradient, có thể lớn hơn.
- Lưu ý đúng đắn: pipeline hiện **không** chọn mô hình theo độ chính xác test (chỉ log). Giữ nguyên nguyên tắc này.

**Cơ sở khoa học.** Huấn luyện/suy luận độ chính xác hỗn hợp (Micikevicius và cộng sự, 2018) cho suy luận fp16 nhanh hơn trên GPU có tensor core mà ít ảnh hưởng độ chính xác; đánh giá không cần gradient nên dùng `inference_mode` và batch lớn. Với nhiễu seed lớn hơn khác biệt giữa các epoch liền kề, đánh giá mỗi epoch không mang thêm thông tin cho quyết định.

**Đề xuất khắc phục.**
1. `eval_every` (mặc định 1 để giữ log cũ; khuyến nghị 5) và luôn đánh giá ở epoch cuối (Phụ lục A5).
2. Đánh giá log dùng `inference_mode` + autocast; **giữ fp32 cho trích feature phục vụ thống kê/drift** (ảnh hưởng độ chính xác prototype) trừ khi kiểm chứng tương đương (cosine giữa feature fp16 và fp32 > 0.9999 và cùng độ chính xác).
3. `eval_batch_size` lớn hơn (ví dụ 128–256) nếu bộ nhớ cho phép.
4. `_compute_class_mean`: một lượt duyệt cho mọi lớp mới.
5. Tích lũy số liệu trên GPU (P01).
Ước tính tiết kiệm: **30–50 phút trên 79.6 phút huấn luyện** nếu phần cố định chủ yếu là đánh giá (suy từ log, cần profile để xác nhận).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P16: giảm chi phí đánh giá và trích thống kê, KHÔNG đổi kết quả huấn luyện.

1. models/RSIAT_adapter.py _init_train: thêm cờ eval_every (mặc định 1) và
   eval_at_end (mặc định true): chỉ gọi _compute_accuracy khi (epoch+1) % eval_every == 0
   hoặc ở epoch cuối. Epoch không đánh giá thì ghi "Test_accy n/a" trong log, giữ định
   dạng dòng log.
2. models/base.py: thêm _compute_accuracy_fast (Phụ lục A5): torch.inference_mode,
   autocast theo runtime_context, tích lũy số đúng trên GPU. Cờ eval_fast (mặc định false)
   chọn hàm mới. KHÔNG dùng autocast cho _extract_vectors/extract_features phục vụ
   prototype, covariance và drift (giữ fp32); thêm cờ stats_autocast (mặc định false).
3. Thêm cờ eval_batch_size_multiplier (mặc định 1) áp lên eval_batch_size của plan.
4. _compute_class_mean: thay vòng lặp tạo DataLoader theo lớp bằng một DataLoader duy
   nhất trên dataset của mọi lớp mới (dùng nhãn thật để phân lớp trong accumulator.update).
   Kết quả (means, covariances) phải khớp bản cũ sai số < 1e-6 trên dữ liệu giả.
5. Thêm đo thời gian tường cho: huấn luyện, đánh giá, thống kê lớp, drift, CA; in tổng kết
   theo task.
6. Test (CPU): (a) mặc định giống hệt bản cũ; (b) với eval_every=5 chỉ có 4 lần gọi
   đánh giá trong 20 epoch và có lần cuối; (c) _compute_accuracy_fast cho cùng số đúng như
   bản cũ trên mô hình giả (fp32); (d) thống kê lớp một lượt khớp bản cũ.
Báo cáo theo mẫu P00 và đưa ra bảng thời gian trước/sau do người dùng đo trên Kaggle.
```

**Nghiệm thu.** Bảng thời gian theo pha trước/sau; độ chính xác cuối và thống kê lớp không đổi ngoài nhiễu seed.

---

## P17 — Tối ưu simulator statevector

**Ưu tiên:** trung bình · **Tác động:** T · **Phụ thuộc:** P07 · **Chi phí:** thấp

**Vấn đề.** [Code] `RealStatevectorCircuit.__call__` (`simulator.py`):
- `zero = basis[(basis & bit) == 0]` và `one = ...` là lập chỉ mục bằng mask boolean, kích thước kết quả phụ thuộc dữ liệu nên trên CUDA gây **đồng bộ host–device**; mỗi cặp (lớp, qubit) hai lần, tức 32 lần mỗi lần gọi mạch (q=8, L=2). Mỗi bước incremental gọi mạch khoảng 6 lần (aligner lên feature, qrel student và teacher, aligner lên prototype, `orth_kernel` hai lần) → cỡ 190 lần đồng bộ mỗi bước.
- Chỉ số, `basis`, `zeros_like` bị tạo lại ở mọi qubit, mọi lần gọi; chuỗi q−1 CNOT là q−1 lần `index_select` thay vì một hoán vị gộp.
- `readout` lặp q lần; `use_compile` chỉ bọc `extract_vector` và `fc`, không bọc simulator.
- Bộ nhớ autograd: nhiều tensor `[B, 2^q]` mỗi cặp (lớp, qubit); với q=12, L=3, B=128 ước tính cỡ 0.45 GB mỗi lần gọi (ước tính thô).

**Bằng chứng tương đương.** [Đo] Bản tối ưu (reshape thay cho `index_select`/`index_copy`, một hoán vị gộp cho cả chuỗi CNOT, readout vector hóa) trùng khớp tuyệt đối với logic gốc trên bốn cấu hình ((q,L) = (4,1), (6,2), (8,2), (8,3) có `cycle`): sai khác state và readout bằng 0 trong numpy. **Chưa chạy bằng torch.**

**Cơ sở khoa học.** Tránh đồng bộ host–device trong vòng lặp nóng là thực hành chuẩn của lập trình CUDA; mô phỏng cổng một qubit bằng reshape trục tương ứng là kỹ thuật tiêu chuẩn của các simulator statevector. Chi phí còn lại tuân theo định luật Amdahl: lợi ích thực tế chỉ có sau khi profile.

**Đề xuất khắc phục.** Cài đặt theo Phụ lục A1; giữ bản cũ thành `RealStatevectorCircuitLegacy` chỉ dùng cho test.

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P17: thay RealStatevectorCircuit bằng bản không dùng mask boolean.

1. qrsiat/quantum/simulator.py: đổi tên lớp hiện tại thành RealStatevectorCircuitLegacy.
   Viết lớp mới RealStatevectorCircuit theo Phụ lục A1: bảng chỉ số cache theo device
   (hoán vị gộp cho chuỗi CNOT và CNOT vòng nếu cycle, xperm, zsign), RY bằng reshape
   (B, -1, 2, 2^qubit) và torch.stack, readout vector hóa. Giữ chữ ký __call__ và readout,
   thêm tham số final_entangler (P07).
2. Cờ simulator_impl ("fast" mặc định sau khi test đạt, "legacy" để so sánh).
3. Test (CPU, bắt buộc): với seed cố định và q in {4,6,8}, L in {1,2,3}, cycle in
   {False,True}, so sánh với Legacy: state, readout và fidelity_kernel bằng torch.allclose
   (atol 1e-6), VÀ gradient theo angles và theta bằng allclose (atol 1e-5).
4. Viết scripts/bench_simulator.py: đo thời gian trung bình forward+backward của hai bản
   cho (q,L,B) trong {(8,2,48),(8,2,128),(12,3,48)} và in số lần đồng bộ CUDA nếu có
   (torch.profiler; trên CPU chỉ in thời gian). Người dùng chạy trên Kaggle.
5. KHÔNG tuyên bố mức tăng tốc; chỉ cung cấp script đo.
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Test tương đương đạt (state, readout, kernel, gradient); `bench_simulator.py` cho số đo thực trên GPU; thời gian mỗi bước giảm hoặc ít nhất không tăng.

---

## P18 — Vệ sinh mã, tài liệu lệch mã và tái lập

**Ưu tiên:** thấp · **Tác động:** K · **Phụ thuộc:** không · **Chi phí:** thấp

**Vấn đề.**
1. [Code] `qrsiat/quantum/losses.py` (`qrel_loss`, `qorth_loss`) không được `StepModule` dùng (nó tự tính inline); `PIPELINE.md` mô tả như thể được dùng. Hai bản có thể lệch nhau (cả hai cùng lỗi không detach softmax).
2. [Code] `displacement_cov` (`base.py:361-384`) là mã chết, gọi `self.cov_computation` chưa thấy định nghĩa, in thời gian bằng `print`; `task_loss_weights` không được dùng; `print(self.radius)` (`base.py:359`) thay vì logging.
3. [Log + Code] `probe` có thể đổi `batch_size` im lặng (lần chạy này giữ nguyên 48). Các loss RS/qrel và centering phụ thuộc kích thước batch, nên đổi batch làm đổi kết quả.
4. [Log] `RuntimePlan` có `cudnn_benchmark=True`: cuDNN chọn thuật toán không xác định giữa các lần chạy, thêm nhiễu run-to-run vào so sánh cặp theo seed (P02).
5. [Code] `orth_epsilon` mặc định 0.2 trong code nhưng lần chạy dùng 0.5; cần ghi rõ trong tài liệu.
6. [Code] Kết quả theo task (độ chính xác lớp cũ/mới) chỉ có trong log văn bản, không có tệp cấu trúc để `aggregate_results.py` (P02) đọc.

**Cơ sở khoa học.** Tái lập thực nghiệm (cùng seed, cùng cấu hình, cùng kết quả) là điều kiện để kiểm định cặp (Bouthillier và cộng sự, 2021) có nghĩa. Mã chết và tài liệu lệch mã là nguồn lỗi kiểm thử giả (test pass trên mã không chạy).

**Prompt giao cho trợ lý.**

```text
Nhiệm vụ P18: dọn dẹp và cải thiện tái lập, KHÔNG đổi kết quả huấn luyện mặc định.

1. qrsiat/quantum/losses.py: hoặc (a) cho StepModule gọi qrel_loss/qorth_loss để chỉ còn
   một cài đặt (sau P06), hoặc (b) xóa file và mọi tham chiếu. Chọn (a) nếu không phá
   giao diện test hiện có. Cập nhật PIPELINE.md cho khớp mã.
2. models/base.py: xóa displacement_cov (và tham chiếu cov_computation) nếu grep không
   thấy ai gọi; thay print bằng logging; xóa task_loss_weights nếu P10 không dùng.
3. models/RSIAT_adapter.py _maybe_probe_batch_size: log rõ khi batch bị giảm và bù bằng
   grad_accum_steps để giữ batch hiệu dụng khi có thể; thêm cờ probe_keep_batch
   (mặc định false) từ chối giảm batch.
4. Thêm cờ cudnn_deterministic (mặc định false): đặt cudnn.benchmark=False và
   cudnn.deterministic=True; log cảnh báo về chi phí tốc độ. Dùng cho ablation (P02).
5. Ghi results.json mỗi task trong <output_dir>/<dataset>/seed_<seed>/ gồm: tổng, theo
   nhóm lớp, old/new accuracy, top-5, thời gian từng pha. Định dạng ổn định để
   aggregate_results.py (P02) đọc.
6. Cập nhật docs: orth_epsilon mặc định, mô tả đúng đường kernel_policy/center_mode đã
   thêm, danh sách cờ mới và giá trị mặc định (bảng trong PIPELINE.md).
7. Test (CPU): grep-based test không còn symbol chết; results.json đọc được bởi
   aggregate_results.py; cờ mới có mặc định giữ hành vi cũ.
Báo cáo theo mẫu P00.
```

**Nghiệm thu.** Không còn mã chết đã nêu; tài liệu khớp mã; có `results.json` cho mỗi task.

---

## Phụ lục A — Mã tham chiếu

Các đoạn dưới đây là điểm xuất phát, **chưa được chạy bằng torch** (sandbox không có torch). Riêng thuật toán của A1 đã được đối chiếu với logic gốc bằng numpy.

### A1 — Simulator không dùng mask boolean (P07, P17)

```python
class RealStatevectorCircuit:
    def __init__(self, n_qubits, layers, *, cycle=False):
        # giữ phần kiểm tra tham số như bản hiện tại
        self.n_qubits, self.layers, self.cycle = n_qubits, layers, cycle
        self._cache = {}

    def _tables(self, device):
        t = self._cache.get(device)
        if t is None:
            q = self.n_qubits
            basis = torch.arange(1 << q)
            perm = basis.clone()                        # hoán vị gộp của cả chuỗi CNOT
            for c in range(q - 1):
                perm = perm[torch.where((basis & (1 << c)) != 0,
                                        basis ^ (1 << (c + 1)), basis)]
            if self.cycle and q > 1:
                perm = perm[torch.where((basis & (1 << (q - 1))) != 0, basis ^ 1, basis)]
            xperm = torch.stack([basis ^ (1 << j) for j in range(q)])
            zsign = torch.stack([1.0 - 2.0 * ((basis & (1 << j)) != 0).float()
                                 for j in range(q)], dim=1)
            t = self._cache[device] = (perm.to(device), xperm.to(device), zsign.to(device))
        return t

    def __call__(self, angles, trainable_angles=None, *, final_entangler=True):
        values = angles.float()
        if trainable_angles is not None:
            values = values + trainable_angles.float().unsqueeze(0)
        perm, _, _ = self._tables(values.device)
        B, q = values.shape[0], self.n_qubits
        cos, sin = torch.cos(values / 2), torch.sin(values / 2)
        state = values.new_zeros(B, 1 << q)
        state[:, 0] = 1.0
        for layer in range(self.layers):
            for qubit in range(q):
                x = state.reshape(B, -1, 2, 1 << qubit)
                a, b = x[:, :, 0], x[:, :, 1]
                c = cos[:, layer, qubit].reshape(B, 1, 1)
                s = sin[:, layer, qubit].reshape(B, 1, 1)
                state = torch.stack((c * a - s * b, s * a + c * b), dim=2).reshape(B, -1)
            if layer < self.layers - 1 or final_entangler:
                state = state.index_select(1, perm)
        return state

    def readout(self, state):
        _, xperm, zsign = self._tables(state.device)
        x = (state.unsqueeze(1) * state[:, xperm]).sum(-1)      # [B, q]
        z = state.square() @ zsign.to(state.dtype)              # [B, q]
        return torch.cat([x, z], dim=1).clamp(-1.0, 1.0)
```

### A2 — Hiệu chỉnh băng thông kernel (P03)

```python
@torch.no_grad()
def calibrate_bandwidth(feature_map, feats, labels=None, target=0.1, grid=None):
    """Chọn gamma sao cho trung vị F của cặp (khác lớp nếu có nhãn) gần `target`."""
    grid = grid if grid is not None else torch.logspace(-3, 2, 26)
    best, best_gap = None, float("inf")
    n = feats.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=feats.device)
    if labels is not None:
        mask = mask & (labels[:, None] != labels[None, :])
    for gamma in grid.tolist():
        feature_map.bandwidth.fill_(gamma)
        state, _ = feature_map(feats)
        k = fidelity_kernel(state)
        gap = abs(k[mask].median().item() - target)
        if gap < best_gap:
            best, best_gap = gamma, gap
    feature_map.bandwidth.fill_(best)
    return best
```

Ghi chú: F không đơn điệu theo γ ở hai đầu (tanh bão hòa), nên dùng lưới thay vì chia đôi; có thể loại các γ khiến tỉ lệ `|tanh(γu)| > 0.99` vượt 20%.

### A3 — Thống kê kernel cho chẩn đoán (P01)

```python
@torch.no_grad()
def kernel_diagnostics(K, labels):
    n = K.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=K.device)
    same = (labels[:, None] == labels[None, :]) & ~eye
    diff = labels[:, None] != labels[None, :]
    qs = torch.tensor([0.05, 0.5, 0.95, 0.99], device=K.device)
    stats = {"F_same": torch.quantile(K[same].float(), qs).tolist(),
             "F_diff": torch.quantile(K[diff].float(), qs).tolist()}
    Y = (labels[:, None] == labels[None, :]).float()
    H = torch.eye(n, device=K.device) - 1.0 / n
    Kc, Yc = H @ K.float() @ H, H @ Y @ H                   # CKA có tâm với ma trận nhãn
    stats["cka"] = ((Kc * Yc).sum() / (Kc.norm() * Yc.norm() + 1e-12)).item()
    return stats
```

### A4 — Cache thừa số Cholesky cho CA (P15)

```python
def _cached_scale_tril(self, c_id):
    cache = self.__dict__.setdefault("_scale_tril_cache", {})
    if c_id not in cache:
        cov = self._make_sampling_covariance(self._class_covs[c_id].to(self._device))
        try:
            cache[c_id] = torch.linalg.cholesky(cov)                  # float32 [D, D]
        except RuntimeError:
            eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
            cache[c_id] = torch.linalg.cholesky(cov + 1e-4 * eye)     # jitter, nên log
    return cache[c_id]

# trong vòng CA:
#   L = self._cached_scale_tril(c_id)
#   m = MultivariateNormal(cls_mean.float(), scale_tril=L)
```

Bộ nhớ: 200 lớp × 768² × 4 byte ≈ 472 MB trên GPU; chấp nhận được với 16 GB.

### A5 — Đánh giá nhanh và chính sách tần suất (P16)

```python
def _should_eval(self, epoch):
    every = int(self.args.get("eval_every", 1))
    return every <= 1 or (epoch + 1) % every == 0 or (epoch + 1) == self.tuned_epochs

@torch.inference_mode()
def _compute_accuracy_fast(self, model, loader):
    model.eval()
    ctx = self.args.get("runtime_context")
    correct = torch.zeros((), device=self._device, dtype=torch.long)
    total = 0
    for _, inputs, targets in loader:
        inputs = inputs.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)
        with (ctx.autocast() if ctx is not None else torch.autocast("cuda", enabled=False)):
            logits = model(inputs)["logits"]
        correct += (logits.argmax(dim=1) == targets).sum()
        total += targets.numel()
    return round(correct.item() * 100.0 / total, 2)
```

### A6 — Co covariance về covariance gộp (P14)

```python
def shrink_covariances(covs, counts, nu=30.0):
    """Σ_c' = ((n_c-1) Σ_c + ν Σ_pool) / (n_c-1+ν), Σ_pool = Σ (n_c-1) Σ_c / Σ (n_c-1)."""
    w = (counts - 1).clamp_min(1).to(covs.dtype)                      # [C]
    pool = (w[:, None, None] * covs).sum(0) / w.sum()
    return (w[:, None, None] * covs + nu * pool) / (w + nu)[:, None, None]
```

Lưu ý: nên tính `pool` từ các lớp cùng hệ tọa độ (cùng task) để không trộn feature của các mạng khác nhau.

### A7 — Drift ổn định số với ESS và băng thông theo trung vị (P12)

```python
@torch.no_grad()
def estimate_drift_stable(emb_old, emb_new, prototypes, sigma2=None, kappa=0.5):
    # emb_old, emb_new: [N, D] (float64); prototypes: [C, D]
    d2 = torch.cdist(emb_old, prototypes).square()                    # [N, C]
    if sigma2 is None:
        sigma2 = kappa * d2.median()                                  # heuristic trung vị
    w = torch.softmax(-d2 / (2.0 * sigma2), dim=0)                    # chuẩn hóa theo mẫu
    ess = 1.0 / w.square().sum(dim=0)                                 # [C], trong [1, N]
    delta = w.T @ (emb_new - emb_old)                                 # [C, D]
    return delta, ess, sigma2
```

---

## Phụ lục B — Danh sách kiểm tra chạy trên Kaggle (theo thứ tự)

1. `scripts/doctor.py` để chẩn đoán phía CPU và cấu hình; chạy `--smoke` để kiểm tra nối dây (không so sánh kết quả với RSIAT).
2. Chạy **một seed với `--diagnostics`** (P01) trên cấu hình hiện tại: xác nhận hoặc bác bỏ dự đoán `L_qrel ≈ 0`, `P(G_top1 > ε) ≈ 0`, F của RS ≈ 0.99.
3. Chạy P16 và P15 (giảm thời gian), ghi bảng thời gian theo pha.
4. Chạy B0, B1, B5, C1 với 3 seed (P02) để quyết định nhánh nào đáng giữ.
5. Chỉ sau đó mới chạy các cải tiến P03–P14 và ma trận ablation đầy đủ.

## Phụ lục C — Tài liệu tham khảo

**Đã xác minh bằng tìm kiếm:**
- S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes. *Exponential concentration in quantum kernel methods.* arXiv:2208.11060; Nature Communications (2024).
- R. Shaydulin, S. M. Wild. *Importance of kernel bandwidth in quantum machine learning.* arXiv:2111.05451 (2021); Phys. Rev. A (2022).
- A. Canatar, E. Peters, C. Pehlevan, S. M. Wild, R. Shaydulin. *Bandwidth enables generalization in quantum kernel models.* arXiv:2206.06686.
- J. Bowles, S. Ahmed, M. Schuld. *Better than classical? The subtle art of benchmarking quantum machine learning models.* arXiv:2403.07059 (2024).
- H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, J. R. McClean. *Power of data in quantum machine learning.* Nature Communications 12, 2631 (2021); arXiv:2011.01938.
- L. Yu, B. Twardowski, X. Liu, L. Herranz, K. Wang, Y. Cheng, S. Jui, J. van de Weijer. *Semantic drift compensation for class-incremental learning.* CVPR 2020; arXiv:2004.00440.
- D. Goswami, Y. Liu, B. Twardowski, J. van de Weijer. *FeCAM: Exploiting the heterogeneity of class distributions in exemplar-free continual learning.* NeurIPS 2023; arXiv:2309.14062.

**Từ trí nhớ, cần kiểm tra lại trước khi trích dẫn:**
- H. Wang et al. *CosFace: Large margin cosine loss for deep face recognition.* CVPR 2018.
- F. Schroff, D. Kalenichenko, J. Philbin. *FaceNet.* CVPR 2015.
- W. Park, D. Kim, Y. Lu, M. Cho. *Relational knowledge distillation.* CVPR 2019.
- C. Cortes, M. Mohri, A. Rostamizadeh. *Algorithms for learning kernels based on centered alignment.* JMLR 2012.
- N. Cristianini, J. Shawe-Taylor, A. Elisseeff, J. Kandola. *On kernel-target alignment.* NIPS 2001.
- D. Garreau, W. Jitkrittum, M. Kanagawa. *Large sample analysis of the median heuristic.* arXiv:1707.07269 (2017).
- T. Salimans, D. Kingma. *Weight normalization.* NIPS 2016. D. Mishkin, J. Matas. *All you need is a good init (LSUV).* ICLR 2016.
- S. Ioffe, C. Szegedy. *Batch normalization.* ICML 2015.
- I. Loshchilov, F. Hutter. *Decoupled weight decay regularization.* ICLR 2019.
- P. Micikevicius et al. *Mixed precision training.* ICLR 2018.
- P. Goyal et al. *Accurate, large minibatch SGD.* arXiv:1706.02677 (2017).
- E. Hu et al. *LoRA.* ICLR 2022.
- O. Ledoit, M. Wolf. *A well-conditioned estimator for large-dimensional covariance matrices.* J. Multivariate Analysis (2004). J. Friedman. *Regularized discriminant analysis.* JASA (1989).
- X. Bouthillier et al. *Accounting for variance in machine learning benchmarks.* MLSys 2021.
- M. Schuld, R. Sweke, J. J. Meyer. *Effect of data encoding on the expressive power of variational quantum-machine-learning models.* Phys. Rev. A (2021). V. Havlíček et al. *Supervised learning with quantum-enhanced feature spaces.* Nature (2019).
- T. Yu et al. *Gradient surgery for multi-task learning.* NeurIPS 2020.
- G. Zhang et al. *SLCA: Slow learner with classifier alignment for continual learning on a pre-trained model.* ICCV 2023.

## Phụ lục D — Giới hạn của tài liệu này

- Các kết luận [Đo] đến từ mô phỏng numpy với chiếu Gaussian i.i.d. ở khởi tạo, không phải từ torch hay GPU, và chưa đo `‖W_q‖` thật. P01 tồn tại để xác nhận hoặc bác bỏ chúng.
- Một seed và không có baseline: không có khẳng định nào ở đây về việc phần lượng tử cải thiện hay làm hại độ chính xác.
- Chưa đọc `qrsiat/stats/drift.py`, `qrsiat/stats/accumulators.py`, `learner_mixin.py`, `base.py` dòng 180–215, `data/`, `trainer.py`; mọi nhận định về chúng chỉ dựa trên tài liệu mô tả hoặc suy luận.
- Các con số tiết kiệm thời gian là ước tính suy từ timestamp của log; không có profiler.
- Các đoạn mã ở Phụ lục A chưa được chạy bằng torch.
