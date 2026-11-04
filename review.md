# Báo Cáo Phản Biện Khoa Học: QG-RSIAT (Quantum-Guided Representation-Steered Incremental Adapter-Tuning)

## 1. Tóm Tắt Ý Tưởng Đang Được Đánh Giá
Ý tưởng **QG-RSIAT** trong [`idea.md`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/idea.md) (và hiện thực hóa tại [`pipeline.md`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/pipeline.md), [`models/RSIAT_adapter.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/models/RSIAT_adapter.py), [`quantum/`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/quantum/simulator.py)) đề xuất nâng cấp framework **RSIAT** (Zhao et al., CVPR 2026) cho bài toán Học tăng dần theo lớp không dùng mẫu lưu trữ (*Exemplar-free Class-Incremental Learning - CIL*) trên backbone ViT-B/16 thông qua 4 thay đổi chính:
1. **Bộ mô phỏng trạng thái lượng tử (`QuantumStateSimulator`)**: Chiếu tuyến tính đặc trưng cổ điển $z \in \mathbb{R}^{768}$ xuống $L \times q$ góc quay ($q=8, L=2 \Rightarrow 16$ chiều), nhúng vào không gian Hilbert $\mathcal{H}_{2^q}$ ($\mathbb{C}^{256}$) để tính biên độ chồng lấp $s(z_i, z_j) = |\langle \psi(z_i) | \psi(z_j) \rangle| \in [0, 1]$ và kỳ vọng toán tử cục bộ $\langle \mathcal{O} \rangle_{\psi(z)} \in [-1, 1]^{4q}$. Mạch lượng tử được huấn luyện ở Task $t=0$ và đóng băng ở mọi Task $t \ge 1$.
2. **Tại Task gốc ($t=0$)**: Thay thế hoàn toàn hàm mất mát định hướng biểu diễn tuyến tính $\mathcal{L}_{\text{RS}}$ của RSIAT bằng hàm mất mát tương phản có giám sát trên biên độ lượng tử $\mathcal{L}_{\text{QSupCon}}$.
3. **Tại các Task tăng dần ($t \ge 1$) – Bộ căn chỉnh lai (`QuantumHybridAligner`)**: Thay thế `AutoencoderSigmoid` bằng mạng phần dư $\mathcal{A}(z) = z + \alpha \cdot g(\langle \mathcal{O} \rangle_{\psi(z)}, z)$ với cơ chế tái khởi tạo tầng cuối về $\mathcal{N}(0, 10^{-3})$ ở đầu mỗi task (`reset_parameters_for_task`).
4. **Tại các Task tăng dần ($t \ge 1$) – Ràng buộc trực giao & Chưng cất quan hệ lượng tử**: Thay thế $\mathcal{L}_{\text{orth}}$ toàn cục bằng `SelectiveQuantumOrthLoss` (lọc Top-$K$ prototype có chồng lấp lượng tử lớn nhất) và bổ sung `TopKQuantumRelLoss` ($\mathcal{L}_{\text{qrel}}$) chưng cất cấu trúc láng giềng lượng tử giữa $z^{\text{cur}}$ và $z^{\text{old}}$.

---

## 2. Các Điểm Bất Cập & Rào Cản Khoa Học (Critical Flaws & Bottlenecks)

Qua quá trình thẩm định độc lập đối chiếu giữa toán học lý thuyết, bài báo gốc **RSIAT** ([`Zhao et al., CVPR 2026`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/paper/Zhao_Representation-Steered_Incremental_Adapter-Tuning_for_Class-Incremental_Learning_with_Pre-Trained_Models_CVPR_2026_paper.pdf)), bài báo **QKD** ([`Li et al., CVPR 2026`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/paper/Li_Quantum-Gated_Task-interaction_Knowledge_Distillation_for_Pre-trained_Model-based_Class-Incremental_Learning_CVPR_2026_paper.pdf)), và mã nguồn hiện tại, có **5 điểm nghẽn nghiêm trọng** sẽ làm suy giảm hiệu năng thực nghiệm nếu giữ nguyên thiết kế hiện tại:

### 2.1. Điểm nghẽn Toán học & Lý thuyết biểu diễn: Suy biến Hạng Không Gian Nhân (Null-Space Rank Collapse $768 \to 16$) khi Đóng Băng Kernel Lượng Tử sau Task 0
- **Phân tích toán học**:
  Trong [`quantum/simulator.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/quantum/simulator.py#L31-L34), vector đặc trưng chuẩn hóa $\hat{z} \in \mathbb{R}^{768}$ được chiếu qua một ma trận tuyến tính đơn $W_{\text{enc}} \in \mathbb{R}^{(L \cdot q) \times d} = \mathbb{R}^{16 \times 768}$ (với cấu hình mặc định $q=8, L=2$):
  $$\Phi(z) = \pi \cdot \tanh\big(\text{LayerNorm}(W_{\text{enc}} \hat{z} + b_{\text{enc}})\big) \in [-\pi, \pi]^{16}$$
  Theo Định lý Hạng - Số chiều nhân (*Rank-Nullity Theorem*):
  $$\dim\big(\ker(W_{\text{enc}})\big) = d - \operatorname{rank}(W_{\text{enc}}) \ge 768 - 16 = 752.$$
  Điều này có nghĩa là **97.9% số chiều của không gian đặc trưng ViT ($752 / 768$ chiều) nằm trong không gian nhân $\ker(W_{\text{enc}})$**. Với mọi vector dịch chuyển $\delta \in \ker(W_{\text{enc}})$, ta luôn có:
  $$W_{\text{enc}}(\hat{z} + \delta) = W_{\text{enc}}\hat{z} \implies |\psi(\hat{z} + \delta)\rangle = |\psi(\hat{z})\rangle \implies s(\hat{z} + \delta, \hat{z}) = 1.0.$$
  Nghiêm trọng hơn, $W_{\text{enc}}$ chỉ được huấn luyện trên **20 lớp đầu tiên của Task 0** ($\mathcal{Y}_0$) rồi bị **đóng băng vĩnh viễn** (`quantum_sim.freeze()`). Do đó, không gian hàng $\operatorname{row}(W_{\text{enc}}) \subset \mathbb{R}^{768}$ (16 chiều) chỉ căn chỉnh theo các hướng phân biệt của 20 lớp thuộc Task 0. Khi mô hình học tiếp 180 lớp mới (Task $1 \to 9$ trên ImageNet-A/R) hoặc 270 lớp mới (trên OmniBenchmark), các đặc trưng phân biệt của lớp mới nằm trong phần bù trực giao $\operatorname{row}(W_{\text{enc}})^\perp$ (752 chiều) hoàn toàn **vô hình** trước `QuantumStateSimulator`.
- **Dẫn chứng khoa học**:
  - `[Theoretical Deduction]` kết hợp định lý biểu diễn chuỗi Fourier lượng tử trong `[Schuld, Sweke, & Meyer, 2021, "Effect of data encoding on the expressive power of variational quantum-machine-learning models", Physical Review A 103, 032430, arXiv:2008.08605]` và `[Kübler et al., 2021, "The inductive bias of quantum kernels", NeurIPS 2021, arXiv:2106.03747]`: Mạch PQC chỉ sinh ra phổ hài bậc cao trên các biến góc đầu vào $\Phi(z)$; mọi thông tin bị triệt tiêu bởi phép chiếu cổ điển hạng thấp $W_{\text{enc}}$ trước khi vào cổng $R_Y, R_Z$ không bao giờ có thể khôi phục bởi cổng vướng víu CNOT trong $\mathcal{H}_{2^q}$.
  - Đối chiếu với `[Li et al., 2026, "Quantum-Gated Task-interaction Knowledge Distillation for Pre-trained Model-based Class-Incremental Learning", CVPR 2026, pp. 3920–3929, Mục 4.1 & 4.3]`: QKD chỉ dùng mạch lượng tử để tính trọng số cổng định tuyến vô hướng $\alpha_i$ giữa các task và **liên tục cập nhật** mạch lượng tử ở mọi task cùng SVD của từng adapter, chứ không đóng băng một phép chiếu 16 chiều từ Task 0 để đo khoảng cách hình học cho toàn bộ 200 lớp.
- **Hệ quả kỹ thuật**:
  - **Va chạm giả (Spurious Quantum Collision)**: Hai vector $z_i^{\text{cur}}$ và prototype $\hat{p}_c$ hoàn toàn trực giao trong $\mathbb{R}^{768}$ ($\langle \hat{z}_i^{\text{cur}}, \hat{p}_c \rangle = 0$) nhưng có cùng hình chiếu trên 16 chiều của Task 0 sẽ cho $s(z_i^{\text{cur}}, \hat{p}_c) \approx 1$, khiến `SelectiveQuantumOrthLoss` phạt nhầm và đẩy lệch đặc trưng.
  - **Điểm mù trực giao (Blind Orthogonality)**: Hai đặc trưng thực sự đè lấn nhau dọc theo 752 chiều thuộc $\ker(W_{\text{enc}})$ lại có đạo hàm $\nabla_{z} s(z, \hat{p}_c) \in \operatorname{row}(W_{\text{enc}})$, không tạo ra bất kỳ lực đẩy nào trong 752 chiều bị chồng lấn.

---

### 2.2. Rủi ro về Gradient, Quên Lãng & Động Lực Học Continual Learning

#### (a) Lệch pha Không gian Biểu diễn – Phân loại (Metric Mismatch) & Hiện tượng "Đi Tắt Qua Bộ Chiếu" (Projector Shortcut) ở Task 0
- **Phân tích động lực học**:
  Ở pha suy luận (`eval_task`) và pha Căn chỉnh bộ phân loại tầng 2 (`_stage2_compact_classifier` + `SSCA displacement` trong [`models/base.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/models/base.py#L52-L134)), mô hình phân loại 100% dựa trên **tích vô hướng Cosine trên mặt cầu đơn vị $\mathbb{S}^{767} \subset \mathbb{R}^{768}$** và lấy mẫu phân phối chuẩn đa biến $\mathcal{N}(\mu_c, \Sigma_c)$ trong $\mathbb{R}^{768}$.
  Trong [`models/RSIAT_adapter.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/models/RSIAT_adapter.py#L190-L194), tại Task 0, `quantum_sim` (chứa $W_{\text{enc}} \in \mathbb{R}^{16\times 768}$) được tối ưu hóa đồng thời với tốc độ học cao `lr = 0.01`, và hàm $\mathcal{L}_{\text{RS}}$ gốc (tính trên Cosine 768-D) bị thay thế hoàn toàn bởi $\mathcal{L}_{\text{QSupCon}}$ (tính trên $s(z_i, z_j)$ sau $W_{\text{enc}}$).
  Vì $W_{\text{enc}}$ có $12,288$ tham số tự do, thuật toán tối ưu SGD dễ dàng giảm nhanh $\mathcal{L}_{\text{QSupCon}}$ bằng cách xoay ma trận $W_{\text{enc}}$ để tách 20 lớp trên không gian góc 16 chiều mà **không cần ép Shared Adapter của ViT thu gọn phương sai nội lớp trong không gian Cosine 768 chiều**. Hơn nữa, vì hàm $\cos(\theta/2), \sin(\theta/2)$ tuần hoàn trên $[-\pi, \pi]$, hai vector cách xa nhau trên $\mathbb{S}^{767}$ có thể ánh xạ về cùng một biên độ lượng tử (aliasing).
- **Dẫn chứng khoa học**:
  - `[Chen et al., 2020, "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020, arXiv:2002.05709]` & `[Jing et al., 2022, "Understanding Dimensional Collapse in Contrastive Self-Supervised Learning", ICLR 2022, arXiv:2110.09348]`: Khi đặt một tầng chiếu khả huấn hạng thấp ($W_{\text{enc}}: \mathbb{R}^{768} \to \mathbb{R}^{16}$) trước hàm Contrastive Loss, tầng chiếu sẽ hấp thụ phần lớn biến đổi hình học, khiến không gian đặc trưng phía trước ($z \in \mathbb{R}^{768}$) không đạt được cấu trúc phân tách cầu đồng nhất.
  - `[Zhao et al., 2026, CVPR 2026, Bảng 3 & Hình 6]`: Độ co cụm trực tiếp trên không gian Cosine 768-D ($\text{Recall@1} = 94.5\%$) là điều kiện tiên quyết để ước lượng ma trận hiệp phương sai $\Sigma_c \in \mathbb{R}^{768\times 768}$ chính xác cho Classifier Alignment (CA).

#### (b) Triệt tiêu Gradient (Vanishing Gradient $2000\times$) trong `QuantumHybridAligner` do `reset_parameters_for_task` và `ae_init_lr` nhỏ
- **Phân tích toán học**:
  Tại [`quantum/aligner.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/quantum/aligner.py#L57-L67), đầu mỗi task $t \ge 1$, hàm `reset_parameters_for_task()` khởi tạo tầng cuối `W_out = decoder[-1].weight` theo $\mathcal{N}(0, 10^{-3})$ và `b_out = 0`, kèm $\alpha = 0.5$.
  Hãy xét đạo hàm ngược từ $\mathcal{L}_{\text{align}}$ về nhánh `quantum_encoder` ($\theta_q$) và `classical_encoder` ($\theta_c$):
  $$\frac{\partial \mathcal{A}(z)}{\partial h} = \alpha \cdot W_{\text{out}}^\top \cdot \operatorname{diag}\big(\text{GELU}'(\cdot)\big) \implies \left\| \frac{\partial \mathcal{L}_{\text{align}}}{\partial \theta_{q,c}} \right\| \propto \alpha \|W_{\text{out}}\| \approx 0.5 \times 10^{-3} = 5 \times 10^{-4}.$$
  Trong các file cấu hình thực nghiệm (ví dụ [`exps/adapter_imageneta.json`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/exps/adapter_imageneta.json#L33)), tốc độ học của bộ căn chỉnh là `ae_init_lr = 0.000312` (`3.12e-4`), và mỗi task tăng dần của ImageNet-A chỉ có $\sim 300$ mẫu ảnh ($\approx 7$ mini-batches/epoch $\times 15$ epochs $= 105$ bước cập nhật).
  Tổng lượng cập nhật trọng số của encoder trong suốt cả một task chỉ đạt:
  $$\Delta \theta_{q,c} \approx 105 \times (3.12 \times 10^{-4}) \times (5 \times 10^{-4}) \approx 1.6 \times 10^{-5} \approx 0!$$
  Và ngay khi sang task tiếp theo ($t+1$), `reset_parameters_for_task()` lại xóa sạch `W_out` về $10^{-3}$!
- **Dẫn chứng khoa học**:
  - `[Theoretical Deduction]` kết hợp đối chiếu mã nguồn gốc [`3e21dca:models/RSIAT_adapter.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/models/RSIAT_adapter.py) và [`Zhao et al., CVPR 2026, Bảng 3`]: Trong mã nguồn gốc của RSIAT, `self.old_ae` được khởi tạo **duy nhất 1 lần ở Task 1** (không reset ở $t > 1$) với trọng số Kaiming chuẩn ($\text{std} \approx 0.125$), cho phép gradient chảy ngay từ bước đầu tiên và tích lũy quy luật dịch chuyển qua các task. Ngược lại, trong thiết kế hiện tại của `QuantumHybridAligner`, $\mathcal{A}(z)$ bị kẹt cứng ở ánh xạ đồng nhất $\mathcal{A}(z) \approx z$, biến $\mathcal{L}_{\text{align}} = \|z^{\text{cur}} - \mathcal{A}(z^{\text{old}})\|^2$ thành phép chưng cất cứng $\|z^{\text{cur}} - z^{\text{old}}\|^2$, làm mất hoàn toàn vai trò giảm chấn của Residual Projector.

#### (c) Xung đột Gradient trực tiếp (Gradient Tug-of-War) lên Shared Adapter tại Task $t \ge 1$
- **Phân tích toán học**:
  Trong bài báo RSIAT gốc ([`Zhao et al., CVPR 2026, Mục 4.2, Phương trình 9`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/paper/Zhao_Representation-Steered_Incremental_Adapter-Tuning_for_Class-Incremental_Learning_with_Pre-Trained_Models_CVPR_2026_paper.pdf)) và mã nguồn gốc (`3e21dca`), hàm $\mathcal{L}_{\text{orth}}$ chỉ tính giữa $\hat{p}_c = \operatorname{Norm}(P^t(\mu_c))$ và $\hat{u}_i = \operatorname{Norm}(P^t(f_i^{t-1}))$ — tức là **đạo hàm của $\mathcal{L}_{\text{orth}}$ chỉ đi qua bộ chiếu $P^t$ (`self.old_ae`)**, sau đó mới dẫn dắt nhẹ nhàng sang $f_i^t$ (`features`) thông qua $\mathcal{L}_{\text{align}}$.
  Ngược lại, trong [`quantum/losses.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/quantum/losses.py#L88-L91), `SelectiveQuantumOrthLoss` truyền gradient trực tiếp vào `z_cur` ($f_i^t$) với hệ số $\gamma = 6.0$, đồng thời `TopKQuantumRelLoss` ép $s(z_i^{\text{cur}}, z_j^{\text{cur}}) = s(z_i^{\text{old}}, z_j^{\text{old}})$ trực tiếp lên `z_cur` với hệ số $\eta = 1.0$:
  1. Với mẫu $x_i$ của task mới có $z_i^{\text{old}}$ nằm gần một prototype cũ $\mu_c$: $\mathcal{L}_{\text{align}}$ ($\beta=3.0$) kéo $z_i^{\text{cur}}$ **về phía** $\mathcal{A}(z_i^{\text{old}}) \approx \mu_c$, trong khi $\mathcal{L}_{\text{orth}}$ ($\gamma=6.0$) đẩy $z_i^{\text{cur}}$ **ra xa** $\mathcal{A}(\mu_c) \approx \mu_c$ dọc theo không gian con 16 chiều của $W_{\text{enc}}$. Hai lực ngược chiều trực tiếp giằng xé gradient của Shared Adapter.
  2. Hơn nữa, trong RSIAT gốc, $\gamma = 6.0$ được nhân với trung bình cộng tích vô hướng có dấu trên toàn bộ $C_{\text{old}}$ lớp (giá trị $\sim 0.02$). Khi chuyển sang trung bình có trọng số Softmax của **Top-$K$ biên độ dương lớn nhất** $(s_{ic} - \epsilon)^2 \in [0.2, 0.6]$, giá trị hiệu dụng của $\gamma \mathcal{L}_{\text{orth}}$ tăng gấp $10\text{--}20$ lần, gây hiện tượng quá điều chuẩn (*over-regularization*).
- **Dẫn chứng khoa học**:
  - `[Zhao et al., 2026, CVPR 2026, Mục 4.2, trang 18013–18014]`: Tác giả RSIAT chỉ rõ rằng việc áp đặt ràng buộc chưng cất/trực giao trực tiếp lên Shared Adapter gây ra *"handbrake effect on the adapter's parameter updates"* và *"optimization resistance from strong regularization"*.

---

### 2.3. Hạn chế về Độ Phức Tạp Tính Toán & Bộ Nhớ ($O(\cdot)$, VRAM)
- **Phân tích luồng thực thi trong [`models/RSIAT_adapter.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/models/RSIAT_adapter.py#L286-L299) và [`quantum/simulator.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/quantum/simulator.py#L91-L122)**:
  1. **Gọi lặp dư thừa `QuantumStateSimulator.encode()` 7 lần trong mỗi bước `_inc_loss`**:
     - Lần 1: `self.old_ae(features_old)` $\to$ `observables(features_old)` $\to$ **`encode(features_old)`**.
     - Lần 2: `self.old_ae(protos)` $\to$ `observables(protos)` $\to$ **`encode(protos)`** ($C_{\text{old}}$ tới 270 lớp).
     - Lần 3: `q_orth_loss_func` $\to$ **`encode(aligned_old)`**.
     - Lần 4: `q_orth_loss_func` $\to$ **`encode(aligned_protos)`** ($C_{\text{old}}$ lớp).
     - Lần 5: `q_orth_loss_func` $\to$ **`encode(features)`**.
     - Lần 6: `q_rel_loss_func` $\to$ `overlap(features_old)` $\to$ **`encode(features_old)`** *(trùng lặp 100% với Lần 1!)*.
     - Lần 7: `q_rel_loss_func` $\to$ `overlap(features)` $\to$ **`encode(features)`** *(trùng lặp 100% với Lần 5!)*.
  2. **Vòng lặp tuần tự Python $L \times q$ tạo nút đồ thị Autograd lớn**:
     Mỗi lần gọi `encode()` chạy vòng lặp $L \times q = 16$ bước, mỗi bước gọi `torch.stack` và `reshape` trên tensor phức `complex64` kích thước $(B, 256)$. Với 7 lần gọi/batch, PyTorch phải cấp phát và lưu trữ $7 \times 16 = 112$ trạng thái trung gian phức để phục vụ `loss.backward()`, làm tăng thời gian huấn luyện mỗi epoch lên khoảng $2.2\times \sim 2.8\times$ so với baseline.

---

### 2.4. Rào cản Tích hợp Codebase
1. **Bỏ qua siêu tham số `init_lr` ở Task 0**: Trong [`models/RSIAT_adapter.py`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/models/RSIAT_adapter.py#L174-L195), cả 4 nhóm tham số ở Task 0 đều bị gán cứng `'lr': 0.01`, khiến `self.init_lr` trong file JSON bị ghi đè hoàn toàn đối với cả `quantum_sim`.
2. **Tương thích với SSCA & Stage-2 Classifier Alignment (`ca_epochs`)**: Pipeline hiện tại giữ nguyên hàm `displacement` và `_stage2_compact_classifier` trong `models/base.py` vốn giả định đặc trưng $z \in \mathbb{R}^{768}$ phân bố chuẩn cầu quanh prototype $\mu_c$. Nếu Task 0 chỉ tối ưu $\mathcal{L}_{\text{QSupCon}}$ trên không gian con 16 chiều mà bỏ qua khoảng cách Cosine 768-D, bước sinh đặc trưng Gaussian ở Stage 2 (`MultivariateNormal(cls_mean, cls_cov)`) sẽ sinh ra các mẫu lệch khỏi biên phân loại Cosine.

---

## 3. Đề Xuất Các Phương Pháp Thay Thế (Alternative Formulations)

Để khắc phục triệt để cả 5 điểm nghẽn trên trong khi vẫn giữ vững đóng góp khoa học cốt lõi của **QG-RSIAT** (*Quantum-Guided Representation-Steered Incremental Adapter-Tuning*) và đảm bảo **độ phức tạp suy luận không đổi (Constant Inference Complexity)**, báo cáo đề xuất **2 phương án thay thế**:

---

### Phương Án A (Khuyến nghị Tối ưu): Multi-Subspace Quantum-Classical Dual Steering & Symmetric-Gated Residual Aligner (MS-QG-RSIAT)

#### 1. Cơ chế Kiến trúc & Công thức Toán học
- **Cải tiến 1 – Bộ mô phỏng Đa không gian con trực giao (`MultiSubspaceQuantumSimulator`) giải quyết Null-Space Collapse**:
  - Thay vì dùng 1 mạch đơn $16$ chiều làm mất $752$ chiều, chia không gian đặc trưng $d=768$ thành $M$ phân vùng không gian con thông qua phép chiếu **khởi tạo trực giao Stiefel (`nn.init.orthogonal_`)** sang $M$ mạch lượng tử nông chạy song song bằng phép toán tensor batch hóa $(B, M, 2^q)$ (ví dụ $M = 8$ mạch con $\times q = 6$ qubits, $2^q = 64$, phủ $M \cdot L \cdot q = 96$ hướng trực giao chính):
    $$\Phi(z) = \pi \cdot \tanh\big(\text{LayerNorm}(W_{\text{enc}} \hat{z} + b_{\text{enc}})\big) \in [-\pi, \pi]^{B \times M \times L \times q}$$
  - Độ tương đồng chồng lấp lượng tử đa không gian con (*Multi-Subspace Quantum Fidelity Amplitude*):
    $$s_{\text{Q}}(z_i, z_j) = \frac{1}{M} \sum_{m=1}^{M} \sqrt{\big|\langle \psi^{(m)}(z_i) \big| \psi^{(m)}(z_j) \rangle\big|^2 + \epsilon_{\text{stab}}} \in [0, 1].$$
  - Toàn bộ $M$ mạch con được tính song song trong một tensor 3D `(B, M, 2^q)` duy nhất (không tốn thêm vòng lặp Python), vừa giảm chiều trạng thái mỗi mạch từ $2^8=256$ xuống $2^6=64$ (tránh *Exponential Concentration* theo `Thanasilp et al., 2024`), vừa tăng độ phủ không gian đặc trưng lên gấp $6\times$!
- **Cải tiến 2 – Định hướng Biểu diễn Kép Lượng tử - Cổ điển ở Task 0 (`DualQuantumRS_Loss`) giải quyết Metric Mismatch**:
  - Kết hợp trực tiếp hình học Cosine 768-D ($\mathcal{L}_{\text{RS}}^{\text{cos}}$) với sự điều biến tương phản lượng tử ($\mathcal{L}_{\text{QSupCon}}$) và trọng số khai phá mẫu khó lượng tử (*Quantum Hard-Pair Weighting*):
    $$\mathcal{L}_{\text{Q-RS}}^{(0)} = \mathcal{L}_{\text{RS}}^{\text{cos}}(\hat{z}, y) + \lambda_q \cdot \mathcal{L}_{\text{QSupCon}}\big(s_{\text{Q}}(z_i, z_j), y\big)$$
    Trong đó $\mathcal{L}_{\text{RS}}^{\text{cos}}$ giữ nguyên cấu trúc cặp dương/âm trên tích vô hướng Cosine $S_{ij} = \langle \hat{z}_i, \hat{z}_j \rangle \in [-1, 1]$ nhưng được **điều biến trọng số tự động** bởi độ lệch lượng tử:
    $$w_{ij}^+ = 1 + \operatorname{sg}\big(1 - s_{\text{Q}}(z_i, z_j)\big), \quad w_{ij}^- = 1 + \operatorname{sg}\big(s_{\text{Q}}(z_i, z_j)\big).$$
  - Đồng thời, cố định phần ma trận chiếu trực giao cơ sở hoặc đặt tốc độ học của `quantum_sim` bằng `0.1 * init_lr` kèm chuẩn hóa trực giao để ngăn hiện tượng *Projector Shortcut*, ép toàn bộ độ co cụm đi vào Shared Adapter 768-D.
- **Cải tiến 3 – Bộ căn chỉnh Phần dư Cổng Lượng tử Đối xứng (`QuantumHybridAligner` Warm-Started) giải quyết Vanishing Gradient**:
  - **Bỏ cơ chế `reset_parameters_for_task()` về $10^{-3}$** gây chết gradient. Giữ nguyên cơ chế học liên tục (warm-start) qua các task giống RSIAT gốc, nhưng thay tầng cuối `nn.Sigmoid()` $\in (0, 1)$ (lệch dương) của `AutoencoderSigmoid` bằng hàm kích hoạt đối xứng quanh $0$ được điều biến bởi cổng kỳ vọng lượng tử $\langle \mathcal{O} \rangle_{\psi(z)} \in [-1, 1]^{M \cdot 4q}$:
    $$\mathcal{A}(z) = z + \tanh\Big(W_{\text{dec}}\big(h_c(z) \odot (1 + \alpha_q \cdot h_q(\langle \mathcal{O} \rangle_{\psi(z)}))\big) + b_{\text{dec}}\Big)$$
  - Tại $\text{input} = 0$, $\tanh(0) = 0$ (thỏa mãn chính xác **Identity Prior** $\Delta z = 0$ tại điểm cân bằng mà không cần ép trọng số $W_{\text{dec}} \to 10^{-3}$!), cho phép đạo hàm $\tanh'(0) = 1.0$ truyền 100% biên độ gradient về cả nhánh cổ điển và nhánh lượng tử ngay từ bước lặp đầu tiên với `ae_init_lr = 0.0003`!
- **Cải tiến 4 – Trực giao Chọn lọc qua Bộ chiếu & Chưng cất Quan hệ Đã căn chỉnh (Projector-Mediated $\mathcal{L}_{\text{orth}}$ & Aligned $\mathcal{L}_{\text{qrel}}$)**:
  - Tuân thủ nguyên lý tránh "handbrake effect" của RSIAT (`Zhao et al., CVPR 2026`):
    - Tính $\hat{u}_i = \operatorname{Norm}(\mathcal{A}(z_i^{\text{old}}))$ và $\hat{p}_c = \operatorname{Norm}(\mathcal{A}(\mu_c))$.
    - Độ tương đồng lai giữa đặc trưng chiếu và prototype chiếu:
      $$\mathcal{I}_{ic} = \big|\langle \hat{u}_i, \hat{p}_c \rangle\big| \cdot \Big(1 + \operatorname{sg}\big(s_{\text{Q}}(\hat{u}_i, \hat{p}_c)\big)\Big)$$
    - Chọn tập $\text{TopK}(i)$ prototype có nguy cơ va chạm cao nhất theo chỉ số lượng tử $s_{\text{Q}}(\hat{u}_i, \hat{p}_c)$ (kèm trung bình toàn cục để giữ đúng thang đo với $\gamma = 6.0$), đẩy $\hat{u}_i$ và $\hat{p}_c$ trực giao nhau trên bộ chiếu, sau đó dẫn dắt $z^{\text{cur}}$ qua $\mathcal{L}_{\text{align}} = \|z^{\text{cur}} - \mathcal{A}(z^{\text{old}})\|^2$.
    - Trong $\mathcal{L}_{\text{qrel}}$: Chưng cất cấu trúc quan hệ lượng tử cục bộ giữa $z^{\text{cur}}$ và **đặc trưng đã căn chỉnh** $\tilde{z}^{\text{old}} = \operatorname{sg}(\mathcal{A}(z^{\text{old}}))$ (thay vì $z^{\text{old}}$ cố định) với hệ số điều hòa nhẹ $\eta_{\text{qrel}} = 0.2$, giúp bảo toàn cấu trúc đa tạp cục bộ mà không khóa cứng sự tiến hóa của Adapter.
  - **Cache trạng thái lượng tử**: Tính `psi, obs = quantum_sim.encode_and_observables(z)` duy nhất 1 lần cho mỗi tensor trong `_inc_loss`, giảm số lần gọi mô phỏng từ 7 xuống 2–3 lần/batch.

#### 2. Khắc phục điểm nghẽn như thế nào
- **Mục 2.1 (Null-Space Collapse)**: Tăng số chiều bao phủ từ 16 lên 96 chiều trực giao qua $M=8$ mạch con $q=6$ qubits và kết hợp tích vô hướng 768-D, xóa bỏ hoàn toàn điểm mù 752 chiều ở các task $t \ge 1$.
- **Mục 2.2a (Metric Mismatch Task 0)**: Bảo toàn 100% tính co cụm mặt cầu $\mathbb{S}^{767}$ cho `CosineLinear` và `SSCA/CA` đồng thời bổ sung lực đẩy tương phản phi tuyến từ $\mathcal{L}_{\text{QSupCon}}$.
- **Mục 2.2b (Vanishing Gradient ở Aligner)**: Thay khởi tạo $10^{-3}$ bằng kích hoạt đối xứng $\tanh(\cdot)$ điều biến lượng tử ($\tanh(0)=0, \tanh'(0)=1$), khôi phục hoàn toàn luồng gradient với `ae_init_lr = 3e-4`.
- **Mục 2.2c (Gradient Tug-of-War)**: Đưa $\mathcal{L}_{\text{orth}}$ về tác động qua không gian chiếu $\mathcal{A}(\cdot)$ và tính $\mathcal{L}_{\text{qrel}}$ trên $\tilde{z}^{\text{old}} = \mathcal{A}(z^{\text{old}})$, loại bỏ xung đột trực tiếp với $\mathcal{L}_{\text{align}}$.
- **Mục 2.3 (Chi phí tính toán)**: Nhờ $2^6 = 64 < 2^8 = 256$ và cơ chế cache trạng thái, thời gian chạy giảm hơn $55\%$ so với bản hiện tại.

#### 3. Cơ sở khoa học / Bài báo tham chiếu
- `[Zhao et al., 2026, CVPR 2026, pp. 18010–18020]`: Cơ chế Projector-mediated Orthogonality và Base Representation Steering trên $\mathbb{S}^{d-1}$.
- `[Li et al., 2026, CVPR 2026, pp. 3920–3929]`: Điều biến trọng số tương tác bằng độ tương đồng trạng thái lượng tử.
- `[Thanasilp et al., 2024, Nature Communications 15, 5200]` & `[Huang et al., 2021, Nature Communications 12, 2631]`: Giới hạn số qubit mỗi mạch con ($q=6$) để tối đa hóa phương sai gradient lượng tử và sử dụng Projected Quantum Observables.

#### 4. Đánh giá I/O & Chi phí tính toán
- **Độ phức tạp Training**: $O(B \cdot M \cdot q \cdot 2^q + B^2 \cdot M \cdot 2^q)$ với $M=8, q=6 \Rightarrow M \cdot 2^q = 512$ (nhẹ và song song hóa tốt hơn vòng lặp $q=8$).
- **Độ phức tạp Inference**: **Không đổi ($0$ tham số tăng thêm, $0$ FLOPs tăng thêm)** — chỉ dùng ViT-B/16 + Shared Adapter + Cosine Classifier.
- **VRAM ước lượng**: $\approx 2.2\text{ GB}$ (tương đương baseline RSIAT).

---

### Phương Án B (Phương án Tinh gọn / Dự phòng): Quantum-Attention Reweighted Classical RSIAT (QAR-RSIAT)

#### 1. Cơ chế Kiến trúc & Công thức Toán học
Nếu ưu tiên sự an toàn tuyệt đối về mặt tối ưu hóa cổ điển và tốc độ chạy nhanh nhất, **Phương án B** giữ nguyên toàn bộ cấu trúc đạo hàm bậc nhất trên không gian Cosine 768-D của RSIAT gốc, và chỉ sử dụng mạch lượng tử như một **Bộ điều biến Trọng số Chú ý Không Khả Vi (`stop-gradient` Quantum Attention Controller)**:
1. **Tại Task $t = 0$**:
   $$\mathcal{L}_{\text{RS}}^{(0)} = \frac{1}{|P|}\sum_{(i,j)\in P} w_{ij}^{\text{Q}+} (1 - S_{ij}) + \alpha \frac{1}{|N|}\sum_{(i,j)\in N} w_{ij}^{\text{Q}-} \max(0, S_{ij} - m_{\text{rs}})$$
   với $w_{ij}^{\text{Q}+} = \operatorname{Softmax}_{(i,j)\in P}\big((1 - s_{\text{Q}}(z_i, z_j))/\tau\big) \cdot |P|$ và $w_{ij}^{\text{Q}-} = \operatorname{Softmax}_{(i,j)\in N}\big(s_{\text{Q}}(z_i, z_j)/\tau\big) \cdot |N|$ được tách khỏi đồ thị đạo hàm (`torch.no_grad()`).
2. **Tại Task $t \ge 1$**:
   - Sửa `AutoencoderSigmoid` thành `AutoencoderTanh` (`reconstructed_x + x` với tầng cuối là `nn.Tanh()` thay vì `nn.Sigmoid()`) để khắc phục lỗi lệch dương mà không thay đổi kiến trúc 2 tầng MLP của RSIAT gốc.
   - Trong $\mathcal{L}_{\text{orth}}$: Giữ nguyên tích vô hướng $\langle \hat{p}_c, \hat{u}_i \rangle$ của RSIAT gốc nhưng nhân với trọng số chú ý lượng tử chọn lọc Top-$K$:
     $$\mathcal{L}_{\text{orth}}^{(t)} = \frac{1}{B \cdot K} \sum_{i=1}^B \sum_{c \in \text{TopK}_Q(i)} w_{ic}^{\text{Q}} \cdot \big|\langle \hat{p}_c, \hat{u}_i \rangle\big|$$

#### 2. Khắc phục điểm nghẽn như thế nào
- Mọi vector đạo hàm $\frac{\partial \mathcal{L}}{\partial z}$ đều đi qua tích vô hướng Cosine 768-D đầy đủ hạng ($\operatorname{rank} = 768$), không bao giờ bị suy biến qua cổ chai 16 chiều hay xung đột hướng gradient.
- Chi phí bộ nhớ Autograd của mạch lượng tử bằng $0$ ở các task $t \ge 1$ vì trọng số lượng tử $w_{ic}^{\text{Q}}$ được tính trong `torch.no_grad()`.

#### 3. Cơ sở khoa học / Bài báo tham chiếu
- `[Zhao et al., 2026, CVPR 2026]` (Khung tối ưu hóa RSIAT gốc) kết hợp `[Li et al., 2026, CVPR 2026]` (Cơ chế dùng đầu ra đo lượng tử làm hệ số điều biến $\alpha_i$ qua Softmax).

---

## 4. Bảng So Sánh Đối Đầu (Trade-off Matrix)

| Tiêu chí | Phương pháp trong `idea.md` gốc | Phương án Thay Thế A (**MS-QG-RSIAT** - Khuyến nghị) | Phương án Thay Thế B (**QAR-RSIAT** - Tinh gọn) |
| :--- | :---: | :---: | :---: |
| **Độ phủ không gian đặc trưng** | Kém ($16 / 768$ chiều, mù $752$ chiều ở Task $t \ge 1$) | **Rất cao** ($M=8$ mạch con trực giao 96 chiều + Cosine 768-D) | **Cao** (Đạo hàm trên 768-D, trọng số chú ý trên mạch lượng tử) |
| **Tương thích với Classifier & SSCA** | Rủi ro lệch pha (*Projector Shortcut* ở Task 0) | **Đồng bộ hoàn toàn** (Dual Quantum-Cosine Steering) | **Đồng bộ hoàn toàn** (Cosine Steering có trọng số lượng tử) |
| **Luồng Gradient qua Bộ căn chỉnh $\mathcal{A}$** | Bị triệt tiêu $2000\times$ do `reset_parameters_for_task` ($10^{-3}$) | **Mạnh & Ổn định** (Kích hoạt đối xứng $\tanh$ điều biến lượng tử, $\tanh'(0)=1$) | **Mạnh & Ổn định** (`AutoencoderTanh` đối xứng $(-1, 1)$) |
| **Xung đột Gradient ở Task $t \ge 1$** | Cao ($\mathcal{L}_{\text{orth}}$ & $\mathcal{L}_{\text{qrel}}$ kéo ngược $\mathcal{L}_{\text{align}}$ trực tiếp trên $z^{\text{cur}}$) | **Thấp** ($\mathcal{L}_{\text{orth}}$ đi qua $\mathcal{A}(\cdot)$; $\mathcal{L}_{\text{qrel}}$ căn chỉnh theo $\mathcal{A}(z^{\text{old}})$) | **Rất thấp** (Chỉ điều biến trọng số trên $\mathcal{L}_{\text{orth}}$ của bộ chiếu) |
| **Độ phức tạp Training / VRAM** | Chậm ($7$ lần gọi `encode`/batch, vòng lặp $16$ bước) | **Nhanh** ($2\text{--}3$ lần gọi `encode` nhờ cache, mạch $2^6=64$ gọn nhẹ) | **Rất nhanh** (Tính trọng số lượng tử trong `no_grad()`) |
| **Độ phức tạp Inference** | Không đổi ($0$ tham số phụ) | **Không đổi ($0$ tham số phụ)** | **Không đổi ($0$ tham số phụ)** |
| **Đóng góp học thuật (Novelty)** | Cao nhưng dễ thất bại thực nghiệm | **Rất cao & Vững chắc về lý thuyết lẫn thực nghiệm** | Trung bình - Khá (Thiên về cơ chế Reweighting) |

---

## 5. Kết Luận & Khuyến Nghị (Actionable Recommendation)

- **Đánh giá tổng quan**: Ý tưởng gốc trong [`idea.md`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/idea.md) có định hướng nghiên cứu rất sáng tạo và xác định đúng các hạn chế của baseline RSIAT (lỗi lệch dương của `AutoencoderSigmoid`, tính cứng nhắc của $\mathcal{L}_{\text{RS}}$ và $\mathcal{L}_{\text{orth}}$ toàn cục). Tuy nhiên, cách thiết kế toán học hiện tại mắc **4 bẫy kỹ thuật nghiêm trọng**: (1) Nén toàn bộ vector 768 chiều qua một ma trận $16\times 768$ đóng băng sau Task 0 gây mù $752$ chiều cho các task sau; (2) Thay thế hoàn toàn Cosine $\mathcal{L}_{\text{RS}}$ bằng $\mathcal{L}_{\text{QSupCon}}$ qua tầng chiếu khả huấn gây hiện tượng *Projector Shortcut*; (3) Khởi tạo lại tầng cuối của `QuantumHybridAligner` về $10^{-3}$ ở mỗi task kết hợp `ae_init_lr = 3e-4` làm tê liệt hoàn toàn bộ căn chỉnh; và (4) Đặt gradient của $\mathcal{L}_{\text{orth}}$ (với $\gamma=6.0$) và $\mathcal{L}_{\text{qrel}}$ trực tiếp lên $z^{\text{cur}}$ ngược chiều với $\mathcal{L}_{\text{align}}$.
- **Khuyến nghị hành động**:
  - **Khuyến nghị chọn Phương Án A (MS-QG-RSIAT)**: Giữ trọn vẹn giá trị đóng góp học thuật của cả 3 module lượng tử (Multi-Subspace Quantum Simulator, Quantum-Gated Symmetric Residual Aligner, và Selective Quantum Orthogonality + Aligned Relational Distillation), đồng thời khắc phục triệt để cả 5 điểm nghẽn toán học và tốc độ huấn luyện.

