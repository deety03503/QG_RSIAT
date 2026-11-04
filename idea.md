# Nghiên Cứu Ý Tưởng: QG-RSIAT (Multi-Subspace Quantum-Guided Representation-Steered Incremental Adapter-Tuning)

> **Trạng thái thẩm định phản biện**: Đã được chuẩn hóa theo **Phương Án A (MS-QG-RSIAT)** sau quá trình phản biện khoa học độc lập. Chi tiết phân tích toán học, dẫn chứng lý thuyết và đánh giá đánh đổi được lưu tại [`review.md`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/review.md).

---

## 1. Giả thuyết khoa học cốt lõi (Core Hypothesis)
- **Phát biểu giả thuyết**: Việc phân rã đặc trưng cổ điển $z \in \mathbb{R}^d$ ($d=768$) của mô hình tiền huấn luyện (ViT-B/16) vào **$M$ không gian con lượng tử trực giao** ($\bigoplus_{m=1}^M \mathcal{H}_{2^q}$, ví dụ $M=8$ mạch con $\times q=6$ qubits phủ $M \cdot L \cdot q = 96$ hướng trực giao Stiefel) cho phép nắm bắt các tương quan hình học phi tuyến bậc cao thông qua **biên độ chồng lấp lượng tử đa không gian con** (*multi-subspace quantum overlap amplitude*) $s_{\text{Q}}(z_i, z_j) = \frac{1}{M}\sum_{m=1}^M |\langle \psi^{(m)}(z_i) | \psi^{(m)}(z_j) \rangle| \in [0, 1]$ và **kỳ vọng quan sát cục bộ** (*projected quantum observables*) $\langle \mathcal{O} \rangle_{\psi(z)} \in [-1, 1]^{M \cdot 4q}$ mà không gặp hiện tượng suy biến hạng không gian nhân (*null-space rank collapse*) hay hội tụ hàm mũ (*exponential concentration*). Khi kết hợp kernel lượng tử đa không gian con đóng băng sau task gốc ($t=0$) với:
  1. **Định hướng biểu diễn kép Lượng tử - Cosine** ($\mathcal{L}_{\text{Q-RS}}$) tại Task $t=0$ (kết hợp trực tiếp hình học cầu $\mathbb{S}^{d-1}$ có trọng số mẫu khó lượng tử với tương phản $\mathcal{L}_{\text{QSupCon}}$),
  2. **Bộ căn chỉnh phần dư cổng lượng tử đối xứng liên nhiệm vụ** ($\mathcal{A}(z) = z + \tanh(\text{Decoder}(h_c(z) \odot (1 + \alpha_q h_q(\langle \mathcal{O} \rangle_{\psi(z)}))))$) đi kèm **ràng buộc trực giao chọn lọc qua bộ chiếu** ($\mathcal{L}_{\text{orth}}$), và
  3. **Chưng cất quan hệ láng giềng lượng tử cục bộ đã căn chỉnh** ($\mathcal{L}_{\text{qrel}}$ giữa $z^{\text{cur}}$ và $\tilde{z}^{\text{old}} = \mathcal{A}(z^{\text{old}})$),
  mô hình có thể giải quyết đồng thời hiện tượng trôi dạt biểu diễn (*feature drift*), lệch pha không gian phân loại (*metric mismatch*), và xung đột gradient (*gradient tug-of-war*) qua các task tăng dần mà **không làm tăng bất kỳ tham số hay chi phí tính toán nào khi suy luận (inference)**.
- **Mục tiêu chính cần đạt được**:
  - Cải thiện độ chính xác trung bình ($\bar{\mathcal{A}}$) và độ chính xác task cuối ($\mathcal{A}_B$) so với baseline RSIAT (CVPR 2026) trên 6 bộ dữ liệu CIL không dùng mẫu lưu trữ (*exemplar-free*): ImageNet-A, ImageNet-R, CIFAR-100, CUB-200, OmniBenchmark, VTAB.
  - Giữ nguyên 100% tốc độ và số lượng tham số ở pha suy luận (`eval_task`) bằng cách chỉ sử dụng backbone ViT + 1 Shared Adapter + Cosine Classifier (loại bỏ hoàn toàn bộ căn chỉnh lượng tử và mạch PQC khi test).

---

## 2. Động lực lý thuyết & Cơ sở khoa học (Theoretical Motivation & Literature)

### Vấn đề của phương pháp hiện tại (Baseline RSIAT Limitations & Naive Quantum Bottlenecks)
Qua đối chiếu giữa bài báo gốc **RSIAT** (Zhao et al., CVPR 2026), mã nguồn gốc (`models/RSIAT_adapter.py`, `utils/toolkit.py`), và báo cáo phản biện [`review.md`](file:///e:/LAB/Code/QG-RSIAT/QG_RSIAT/review.md), có **5 điểm nghẽn lý thuyết và thực nghiệm** cần giải quyết:
1. **Hạn chế của Kernel Cosine tuyến tính và Margin phẳng ở Task gốc ($t=0$) & Bẫy *Projector Shortcut* nếu bỏ Cosine 768-D**:
   - Baseline RSIAT sử dụng hàm mất mát cặp tuyến tính $\mathcal{L}_{\text{RS}}$ trên tích vô hướng Euclid chuẩn hóa $S_{ij} = \langle \hat{z}_i, \hat{z}_j \rangle \in [-1, 1]$ với đạo hàm phẳng, thiếu cơ chế tự động ưu tiên các cặp mẫu khó (*hard-positive / hard-negative mining*).
   - Tuy nhiên, nếu thay thế hoàn toàn $\mathcal{L}_{\text{RS}}$ trên $\mathbb{S}^{767}$ bằng $\mathcal{L}_{\text{QSupCon}}$ sau một tầng chiếu khả huấn hạng thấp, bộ tối ưu sẽ đi tắt qua tầng chiếu (*projector shortcut*, Chen et al., ICML 2020; Jing et al., ICLR 2022) mà không ép Shared Adapter 768-D co cụm trên mặt cầu đơn vị phục vụ `CosineLinear` và `SSCA/CA`. Do đó, bắt buộc phải kết hợp **Dual Quantum-Cosine Steering**.
2. **Suy biến Hạng Không Gian Nhân (*Null-Space Collapse*) khi dùng 1 mạch PQC đơn**:
   - Nếu chiếu $z \in \mathbb{R}^{768}$ xuống 1 mạch đơn $L \times q = 16$ góc quay rồi đóng băng sau Task 0, không gian nhân $\ker(W_{\text{enc}})$ có chiều $\ge 752$ (chiếm $97.9\%$ số chiều), làm mô hình bị "mù" trước các đặc trưng mới xuất hiện ở Task $t \ge 1$ (Schuld et al., PRA 2021; Kübler et al., NeurIPS 2021). Việc phân rã thành **$M=8$ mạch con trực giao ($q=6$ qubits/mạch)** vừa nâng số chiều bao phủ trực giao lên $6\times$ ($96$ hướng Stiefel), vừa giảm chiều Hilbert mỗi mạch con xuống $2^6 = 64$ để triệt tiêu hoàn toàn hiện tượng *Exponential Concentration* (Thanasilp et al., Nat. Commun. 2024).
3. **Lỗi lệch dương (*Positivity Bias*) của `AutoencoderSigmoid` và Bẫy Triệt tiêu Gradient khi Reset Trọng số về $10^{-3}$**:
   - Trong `utils/toolkit.py`, `AutoencoderSigmoid` dùng `reconstructed_x + x` với tầng cuối là `nn.Sigmoid()` $\in (0, 1)$, ép dịch chuyển phần dư $\Delta z > 0$ (lệch dương).
   - Ngược lại, nếu khởi tạo tầng cuối về $\mathcal{N}(0, 10^{-3})$ và xóa lại ở đầu mỗi task (`reset_parameters_for_task`), kết hợp với `ae_init_lr = 0.0003` và số bước lặp ít ($\sim 100$ iterations/task), đạo hàm ngược về encoder bị nhân với $10^{-3}$ gây **triệt tiêu gradient $2000\times$**, làm tê liệt bộ căn chỉnh. Giải pháp chuẩn xác về mặt toán học là sử dụng hàm kích hoạt đối xứng $\tanh(\cdot) \in (-1, 1)$ điều biến bởi cổng kỳ vọng lượng tử ($\tanh(0)=0$ bảo toàn Identity Prior tại cân bằng, trong khi đạo hàm $\tanh'(0)=1$ giữ nguyên 100% biên độ gradient) và duy trì học liên tục (*warm-start*) qua các task.
4. **Quá ràng buộc (*Over-regularization*) và Xung đột Gradient (*Gradient Tug-of-War*) trong $\mathcal{L}_{\text{orth}}$**:
   - Baseline RSIAT ép trực giao đồng đều trên toàn bộ prototype cũ làm giảm tính dẻo (*plasticity*, Bảng 3 của Zhao et al., CVPR 2026). Tuy nhiên, nếu truyền trực tiếp đạo hàm của $\mathcal{L}_{\text{orth}}$ (với $\gamma=6.0$) vào đặc trưng hiện tại $z^{\text{cur}}$, nó sẽ kéo ngược chiều trực tiếp với $\mathcal{L}_{\text{align}}$ (gây *handbrake effect*). Do đó, $\mathcal{L}_{\text{orth}}$ cần được định hướng thông qua không gian chiếu $\mathcal{A}(\cdot)$ với trọng số chọn lọc lượng tử Top-$K$ được chuẩn hóa đúng thang đo với $\gamma=6.0$.
5. **Thiếu cơ chế bảo toàn cấu trúc hình học lân cận đã căn chỉnh (*Aligned Local Relational Topology*)**:
   - $\mathcal{L}_{\text{align}}$ chỉ ràng buộc khoảng cách từng điểm (*point-wise MSE*). Bổ sung chưng cất quan hệ láng giềng lượng tử cục bộ Top-$K$ ($\mathcal{L}_{\text{qrel}}$) giữa $z^{\text{cur}}$ và **đặc trưng cũ đã căn chỉnh** $\tilde{z}^{\text{old}} = \operatorname{sg}(\mathcal{A}(z^{\text{old}}))$ giúp bảo toàn cấu trúc đa tạp cục bộ mà không khóa cứng Adapter vào không gian cũ chưa dịch chuyển.

### Tài liệu tham khảo chính
- **Zhao et al. (2026)**, *"Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models"*, CVPR 2026, pp. 18010–18020. (Baseline trực tiếp).
- **Li et al. (2026)**, *"Quantum-Gated Task-interaction Knowledge Distillation for Pre-trained Model-based Class-Incremental Learning"*, CVPR 2026, pp. 3920–3929. (Cơ sở cho cơ chế điều biến tương tác bằng độ tương đồng trạng thái lượng tử trong PTM-based CIL).
- **Liu et al. (2025)**, *"Quantum Relational Knowledge Distillation"*, arXiv:2508.13054. (Cơ sở lý thuyết cho chưng cất cấu trúc quan hệ bằng Quantum State Fidelity lúc train và suy luận hoàn toàn cổ điển).
- **Thanasilp et al. (2024)**, *"Exponential concentration in quantum kernel methods"*, Nature Communications 15, 5200. (Định lý giới hạn số qubit mỗi mạch con $q=6$ để tránh suy biến phương sai kernel).
- **Huang et al. (2021)**, *"Power of data in quantum machine learning"*, Nature Communications 12, 2631. (Cơ sở lý thuyết của Projected Quantum Kernels qua kỳ vọng toán tử cục bộ $\langle \mathcal{O} \rangle_{\psi(z)}$).
- **Schuld, Sweke, & Meyer (2021)**, *"Effect of data encoding on the expressive power of variational quantum-machine-learning models"*, Physical Review A 103, 032430. (Cơ sở của thiết kế mã hóa đa không gian con trực giao tránh suy biến hạng).
- **Khosla et al. (2020)**, *"Supervised Contrastive Learning"*, NeurIPS 2020. (Cơ sở của thành phần $\mathcal{L}_{\text{QSupCon}}$).

---

## 3. Ràng buộc & Đặc tả Input/Output (Input/Output Specifications)

- **Input**:
  - Đặc trưng đầu ra từ backbone ViT-B/16 kèm Shared Adapter: $z^{\text{cur}} = \phi(x; A^t) \in \mathbb{R}^{B \times d}$ và $z^{\text{old}} = \phi(x; A^{t-1}) \in \mathbb{R}^{B \times d}$ ($B=48, d=768$).
  - Nhãn lớp của batch hiện tại: $y \in \mathbb{Z}^B$.
  - Tập prototype của các lớp cũ đã tích lũy: $\{\mu_c\}_{c=0}^{\mathcal{C}_{\text{old}}-1} \in \mathbb{R}^{\mathcal{C}_{\text{old}} \times d}$.
- **Không gian lượng tử đa phân vùng (`MultiSubspaceQuantumSimulator`)**:
  - Số mạch con song song: $M = 8$ (mặc định), số qubit mỗi mạch con: $q = 6$ (chiều Hilbert mỗi mạch con $2^q = 64$, tổng chiều biểu diễn phức $M \cdot 2^q = 512$), độ sâu mỗi mạch: $L = 2$ tầng Data Re-uploading.
  - Khởi tạo trực giao Stiefel: Ma trận chiếu $W_{\text{enc}} \in \mathbb{R}^{(M \cdot L \cdot q) \times d}$ được khởi tạo trực giao (`nn.init.orthogonal_`) để bảo toàn khoảng cách toàn cục trên $\mathbb{R}^{768}$.
  - Trạng thái lượng tử đa không gian con: $|\psi(z)\rangle \in \mathbb{C}^{B \times M \times 2^q}$ chuẩn hóa $\|\psi^{(m)}(z)\|_2 = 1, \forall m \in \{1,\dots,M\}$.
  - Biên độ chồng lấp trung bình đa không gian con:
    $$s_{\text{Q}}(z_i, z_j) = \frac{1}{M} \sum_{m=1}^M \sqrt{\big|\langle \psi^{(m)}(z_i) \big| \psi^{(m)}(z_j) \rangle\big|^2 + \epsilon_{\text{stab}}} \in [0, 1], \quad \epsilon_{\text{stab}} = 10^{-8}.$$
  - Kỳ vọng quan sát cục bộ: $\langle \mathcal{O} \rangle_{\psi(z)} \in [-1, 1]^{B \times (M \cdot 4q)}$ (gồm $\langle \sigma_z \rangle, \langle \sigma_x \rangle, \langle \sigma_y \rangle, \langle \sigma_z^{(k)} \sigma_z^{(k+1)} \rangle$ trên tất cả $M$ mạch con, tổng cộng $4 M q = 192$ đặc trưng kỳ vọng lượng tử).
- **Output & Hàm Mất Mát**:
  - **Tại Task gốc ($t = 0$) – Dual Quantum-Cosine Representation Steering**:
    $$\mathcal{L}_{\text{total}}^{(0)} = \mathcal{L}_{\text{cos}} + \lambda_{\text{rs}}(e) \cdot \Big( \mathcal{L}_{\text{RS}}^{\text{Q-weighted}}(\hat{z}, y) + \lambda_q \cdot \mathcal{L}_{\text{QSupCon}}\big(s_{\text{Q}}(z_i, z_j), y\big) \Big)$$
    trong đó $\mathcal{L}_{\text{RS}}^{\text{Q-weighted}}$ điều biến trực tiếp khoảng cách Cosine 768-D bằng trọng số mẫu khó lượng tử $w_{ij}^+ = 1 + \operatorname{sg}(1 - s_{\text{Q}}(z_i, z_j))$ và $w_{ij}^- = 1 + \operatorname{sg}(s_{\text{Q}}(z_i, z_j))$.
  - **Tại các Task tăng dần ($t \ge 1$)**:
    - Đặc trưng cũ đã căn chỉnh qua cổng lượng tử đối xứng:
      $$\tilde{z}^{\text{old}} = \mathcal{A}(z^{\text{old}}) = z^{\text{old}} + \tanh\Big(W_{\text{dec}}\big(h_c(z^{\text{old}}) \odot (1 + \alpha_q \cdot h_q(\langle \mathcal{O} \rangle_{\psi(z^{\text{old}})}))\big) + b_{\text{dec}}\Big) \in \mathbb{R}^{B \times d}.$$
    - Tổng hàm mất mát:
      $$\mathcal{L}_{\text{total}}^{(t)} = \mathcal{L}_{\text{cos}} + \beta \mathcal{L}_{\text{align}}(z^{\text{cur}}, \tilde{z}^{\text{old}}) + \gamma \mathcal{L}_{\text{orth}}(\tilde{z}^{\text{old}}, \mathcal{A}(\mu)) + \eta \mathcal{L}_{\text{qrel}}\big(z^{\text{cur}}, \operatorname{sg}(\tilde{z}^{\text{old}})\big).$$
- **Ràng buộc thuật toán**:
  1. **Exemplar-free**: Tuyệt đối không lưu trữ mẫu ảnh của các task cũ.
  2. **Constant Inference Complexity**: Toàn bộ simulator lượng tử và bộ căn chỉnh $\mathcal{A}$ chỉ tham gia trong pha huấn luyện (`_train`). Ở pha suy luận (`eval_task`), mô hình chỉ dùng $\phi(x; A^t)$ và tầng phân loại tuyến tính Cosine.
  3. **Stationary Multi-Subspace Quantum Metric**: Tham số của mạch lượng tử được tinh chỉnh nhẹ (`lr = 0.1 * init_lr`) ở Task 0 ($t=0$) quanh khởi tạo trực giao Stiefel và **đóng băng hoàn toàn** (`requires_grad = False`) ở mọi task $t \ge 1$.
  4. **State Caching & Vectorization**: Mọi phép toán mô phỏng lượng tử trên $M$ mạch con được vector hóa hoàn toàn trên tensor 3D `(B, M, 2^q)` và lưu đệm (`encode_and_observables`) để tránh tính lặp trạng thái lượng tử trong cùng một mini-batch.

---

## 4. Lợi thế kỳ vọng so với Baseline (Expected Gains)
- **Định lượng**:
  - Tăng độ chính xác trung bình ($\bar{\mathcal{A}}$) thêm $+0.8\% \sim +2.0\%$ và độ chính xác task cuối ($\mathcal{A}_B$) thêm $+1.0\% \sim +2.5\%$ so với baseline RSIAT trên các benchmark khó như **ImageNet-A (B0I20)** và **ImageNet-R (B0I20)**.
  - Giảm thời gian huấn luyện hơn $55\%$ so với thiết kế PQC đơn $q=8$ chưa cache nhờ mô phỏng song song $M=8, q=6$ ($2^6=64$) và loại bỏ các lời gọi `encode()` dư thừa.
- **Định tính**:
  - Bảo toàn đồng thời cấu trúc cầu chuẩn 768-D cho `CosineLinear` / `SSCA` và biên phân tách phi tuyến bậc cao trên không gian Hilbert đa phân vùng.
  - Khắc phục hoàn toàn lỗi lệch dương $(0, 1)$ của `AutoencoderSigmoid` bằng hàm kích hoạt đối xứng $\tanh \in (-1, 1)$ điều biến lượng tử mà vẫn giữ nguyên $100\%$ biên độ đạo hàm $\tanh'(0)=1$.
  - Loại bỏ hiện tượng xung đột gradient trực tiếp lên Shared Adapter nhờ cơ chế trực giao chọn lọc qua bộ chiếu $\mathcal{A}(\cdot)$ và chưng cất quan hệ trên đặc trưng đã căn chỉnh $\operatorname{sg}(\mathcal{A}(z^{\text{old}}))$.

---

## 5. Rủi ro & Lưu ý khi Hiện thực hóa (Risks & Implementation Notes)
1. **Bảo toàn tính trực giao của $W_{\text{enc}}$ ở Task 0**:
   - Khởi tạo $W_{\text{enc}}$ bằng `nn.init.orthogonal_` và sử dụng tốc độ học nhỏ (`0.1 * init_lr`) ở Task 0 để ngăn $W_{\text{enc}}$ suy biến về một không gian con hẹp của riêng 20 lớp đầu tiên.
2. **Đạo hàm an toàn tại điểm trực giao ($s_{\text{Q}} \to 0$) và Anchors đơn độc**:
   - Luôn cộng $\epsilon_{\text{stab}} = 10^{-8}$ bên trong căn bậc hai khi tính biên độ chồng lấp lượng tử trên từng mạch con $m \in \{1,\dots,M\}$, và lọc bỏ các anchor không có mẫu cùng lớp trong batch (`pos_count > 0`) khi tính $\mathcal{L}_{\text{QSupCon}}$.
3. **Không xóa trắng trọng số Bộ căn chỉnh $\mathcal{A}(z)$ về $10^{-3}$ ở mỗi Task**:
   - Không gọi `nn.init.normal_(..., std=1e-3)` ở đầu mỗi task tăng dần vì với `ae_init_lr = 0.0003` và số batch ít, trọng số $10^{-3}$ sẽ chặn đứng gradient truyền ngược về encoder. Thay vào đó, sử dụng nhánh decoder kết thúc bằng `nn.Tanh()` được khởi tạo Kaiming chuẩn và duy trì học liên tục (*warm-start*) qua các task giống như `self.old_ae` của RSIAT gốc.
4. **Định chuẩn thang đo của $\mathcal{L}_{\text{orth}}$ và $\mathcal{L}_{\text{qrel}}$**:
   - Đảm bảo giá trị trung bình của $\mathcal{L}_{\text{orth}}$ tương thích trực tiếp với hệ số $\gamma$ trong các file cấu hình JSON (`gamma = 6.0` trên ImageNet-A, `0.1–1.0` trên các tập khác), và đặt hệ số mặc định của $\mathcal{L}_{\text{qrel}}$ ở mức điều hòa êm (`eta_qrel = 0.2`) để hỗ trợ $\mathcal{L}_{\text{align}}$ mà không gây cứng nhắc cho Shared Adapter.
