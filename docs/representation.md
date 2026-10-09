# Quantum-Relational Representation Steering cho học tăng dần theo lớp

## Tóm tắt ý tưởng

Phương pháp dùng trạng thái lượng tử mô phỏng để biểu diễn feature ảnh và dùng fidelity giữa các trạng thái làm độ tương đồng. Cùng phép đo fidelity này tạo tín hiệu cho ba việc: định hình quan hệ lớp ở base task, giữ quan hệ giữa các mẫu qua task, và tìm các prototype lớp cũ dễ bị nhầm với mẫu mới. Ở incremental task, một residual aligner lai cổ điển-lượng tử ánh xạ feature cũ sang không gian hiện tại; readout Pauli-X/Z của mạch tạo phần cập nhật residual.

Mạch chạy bằng statevector khả vi trên máy cổ điển và chỉ cần trong huấn luyện. Dữ liệu ảnh cũ không được lưu; chỉ thống kê lớp như prototype và covariance được giữ để tính loss bảo vệ lớp cũ và cập nhật classifier.

## 1. Bài toán và ký hiệu

Cho chuỗi task \(\mathcal{D}_1,\ldots,\mathcal{D}_T\), mỗi task cung cấp ảnh của một tập lớp mới. Tại task \(t\), mạng hiện tại trích xuất feature \(f_i^t\in\mathbb{R}^d\). Với \(t>1\), mạng của task trước được đóng băng làm teacher và tạo feature tham chiếu \(f_i^{t-1}\) cho cùng ảnh.

Gọi \(q\) là số qubit, \(L\) là số lớp mạch, \(\psi(z)\) là state được mã hóa từ feature \(z\), và \(\operatorname{sg}(\cdot)\) là phép dừng gradient. Backbone và classifier tạo feature/logit cho phân loại; phần mới của phương pháp là ánh xạ feature vào statevector và các mục tiêu lượng tử được mô tả dưới đây.

## 2. Nhúng feature vào không gian lượng tử

### 2.1. Chiếu feature thành góc quay

Với feature đầu vào \(z\in\mathbb{R}^d\), tùy chọn centering trừ đi tâm \(c\). Một lớp tuyến tính đưa feature về \(qL\) giá trị, sau đó reshape thành \(L\) lớp, mỗi lớp có \(q\) góc quay:

\[
u=W_q(z-c)+b_q\in\mathbb{R}^{qL},\qquad
\alpha_{\ell j}=\pi\tanh(u_{\ell j})+\theta_{\ell j},
\]

với \(\ell=1,\ldots,L\), \(j=1,\ldots,q\). \(W_q,b_q\) là tham số chiếu cổ điển; \(\theta\in\mathbb{R}^{L\times q}\) là góc quay trainable của mạch. Hàm tanh giới hạn phần góc do feature sinh ra vào khoảng \((-\pi,\pi)\). Khi centering bật, \(c\) là trung bình feature của batch; trong phép so sánh hai tập feature, cùng một tâm trung bình của hai tập được dùng cho cả hai phía.

### 2.2. Chuẩn bị statevector

Mỗi mẫu bắt đầu ở trạng thái cơ sở \(\lvert 0\rangle^{\otimes q}\), tương ứng vector đơn vị có \(2^q\) phần tử. Ở mỗi lớp \(\ell\):

1. Áp dụng cổng quay \(R_Y(\alpha_{\ell j})\) lên từng qubit \(j\).
2. Entangle các qubit bằng chuỗi CNOT lân cận \(0\to1,1\to2,\ldots,q-2\to q-1\).

State mã hóa cuối cùng là:

\[
\lvert\psi(z)\rangle=
U_L(\alpha_L)\cdots U_1(\alpha_1)
\lvert0\rangle^{\otimes q}\in\mathbb{R}^{2^q}.
\]

Do mạch chỉ dùng \(R_Y\) và CNOT, biên độ state là số thực. Statevector được chuẩn hóa bởi tính unitary của mạch. Mã nguồn mô phỏng state chính xác, không lấy mẫu shot từ phần cứng lượng tử.

### 2.3. Readout ở base task: fidelity giữa hai state

Ở base task, nhánh quantum representation-steering không đọc từng qubit thành vector kỳ vọng Pauli. Nó giữ statevector và đọc quan hệ của hai mẫu bằng fidelity:

\[
F_{ij}=\left|\langle\psi(f_i^1)\mid\psi(f_j^1)\rangle\right|^2
=\left((\psi_i)^\top\psi_j\right)^2\in[0,1].
\]

Vì vậy readout của base task là một số cho mỗi cặp mẫu, tạo thành ma trận \(F\in\mathbb{R}^{B\times B}\). Fidelity bằng 1 khi hai state trùng nhau và nhỏ khi chúng gần trực giao. Đây là phép so sánh statevector chính xác trong simulator, không phải một phép đo Pauli riêng lẻ. Ma trận fidelity được đưa trực tiếp vào loss positive/negative ở phần tiếp theo.

## 3. Base task: quantum representation-steering

Với nhãn \(y_i\), tạo mask cặp cùng lớp và khác lớp:

\[
M^+_{ij}=\mathbb{I}[y_i=y_j,\ i\ne j],\qquad
M^-_{ij}=\mathbb{I}[y_i\ne y_j].
\]

Loss positive kéo fidelity của cặp cùng lớp về 1; loss negative phạt fidelity của cặp khác lớp nếu vượt margin \(m_{\mathrm{RS}}\):

\[
\mathcal{L}_{\mathrm{pos}}=
\frac{\sum_{ij}M^+_{ij}[1-F_{ij}]_+}
{\sum_{ij}M^+_{ij}+\epsilon},\qquad
\mathcal{L}_{\mathrm{neg}}=
\frac{\sum_{ij}M^-_{ij}[F_{ij}-m_{\mathrm{RS}}]_+}
{\sum_{ij}M^-_{ij}+\epsilon},
\]

\[
\mathcal{L}_{\mathrm{QRS}}=
\mathcal{L}_{\mathrm{pos}}+\alpha\mathcal{L}_{\mathrm{neg}}.
\]

Ở đây \([a]_+=\max(0,a)\), \(\alpha\) cân bằng cặp âm, và \(\epsilon\) tránh chia cho 0 nếu batch thiếu một loại cặp. Hệ số loss tăng dần để nhánh quantum không lấn át phân loại ở đầu task:

\[
\lambda_{\mathrm{QRS}}(e)=\lambda_{\mathrm{QRS}}^{\max}
\min\left(1,\frac{e}{E_w}\right),\qquad
\mathcal{L}^{1}=\mathcal{L}_{\mathrm{cls}}^{1}
+\lambda_{\mathrm{QRS}}(e)\mathcal{L}_{\mathrm{QRS}}.
\]

\(e\) là epoch, \(E_w\) là độ dài warm-up. \(\mathcal{L}_{\mathrm{cls}}\) giữ nhiệm vụ phân loại nhãn; \(\mathcal{L}_{\mathrm{QRS}}\) định hình cấu trúc theo cặp bằng fidelity quantum.

## 4. Incremental task: ba tín hiệu quantum

Ở task \(t>1\), cùng ảnh mới đi qua mạng hiện tại và teacher đóng băng để thu \(f_i^t\) và \(f_i^{t-1}\). Các nhánh dưới đây dùng thông tin này theo ba cách bổ sung nhau.

### 4.1. Residual aligner và readout Pauli-X/Z

Để ánh xạ feature cũ vào hệ tọa độ mới, residual aligner dùng một projection cổ điển xuống góc mạch, statevector circuit và một readout Pauli:

\[
v=W_{\mathrm{down}}(f-c)+b_{\mathrm{down}},\qquad
\alpha_{\ell j}=\pi\tanh(v_{\ell j})+\theta_{\ell j},\qquad
r(f)=\operatorname{Readout}_{X,Z}(\psi(f)).
\]

Với state thực \(\psi\), readout trả về hai kỳ vọng trên mỗi qubit:

\[
r_j^X=\langle\psi|X_j|\psi\rangle
=\sum_{s=0}^{2^q-1}\psi_s\psi_{s\oplus 2^j},
\]

\[
r_j^Z=\langle\psi|Z_j|\psi\rangle
=\sum_{s=0}^{2^q-1}(-1)^{b_j(s)}\psi_s^2,
\qquad
r(f)=[r_1^X,\ldots,r_q^X,r_1^Z,\ldots,r_q^Z]\in[-1,1]^{2q}.
\]

\(s\) là chỉ số computational-basis, \(b_j(s)\) là bit của qubit \(j\) trong chỉ số đó, và \(\oplus\) là XOR. Kỳ vọng \(X_j\) đo giao thoa giữa hai biên độ chỉ khác bit \(j\); kỳ vọng \(Z_j\) là chênh lệch xác suất qubit ở 0 và 1. Readout Pauli-Y không cần thiết vì state và cổng trong simulator là thực.

Vector \(r(f)\) được chiếu ngược thành hiệu chỉnh \(d\) chiều:

\[
P^t(f)=f+W_{\mathrm{up}}r(f),\qquad W_{\mathrm{up}}^{(0)}=0.
\]

Khởi tạo \(W_{\mathrm{up}}\) bằng 0 đảm bảo \(P^t(f)=f\) ở đầu task; aligner chỉ học phần hiệu chỉnh cần thiết. Loss căn chỉnh giữ feature hiện tại gần feature teacher sau phép chiếu:

\[
\mathcal{L}_{\mathrm{align}}^t=
\frac{1}{B}\sum_{i=1}^{B}
\left\lVert f_i^t-P^t(f_i^{t-1})\right\rVert_2^2.
\]

### 4.2. Quantum relational loss

Tạo hai ma trận fidelity trong batch: ma trận hiện tại từ feature student và ma trận tham chiếu từ feature teacher. Kernel student có thể học; một bản kernel teacher được chụp ở đầu task và đóng băng để mục tiêu không trôi trong lúc tối ưu:

\[
K^t_{ij}=|\langle\psi_{\vartheta_t}(f_i^t)
\mid\psi_{\vartheta_t}(f_j^t)\rangle|^2,
\]

\[
K^{t-1}_{ij}=|\langle\psi_{\vartheta_{t-1}}(f_i^{t-1})
\mid\psi_{\vartheta_{t-1}}(f_j^{t-1})\rangle|^2.
\]

Loss qrel khớp các quan hệ trong hai batch:

\[
\mathcal{L}_{\mathrm{qrel}}^t=
\frac{1}{B^2}\left\lVert
K^t-\operatorname{sg}(K^{t-1})\right\rVert_F^2.
\]

Qrel không buộc feature mới phải bằng feature cũ từng chiều; nó giữ cặp nào tương tự hoặc khác nhau theo hình học fidelity. Trong cấu hình quantum đầy đủ, cả hai ma trận đều dùng feature map quantum.

### 4.3. Trực giao fidelity chọn lọc với prototype cũ

Sau mỗi task, giữ prototype \(\mu_c\) cho lớp đã học. Trong task mới, prototype cũ được ánh xạ qua residual aligner. Quantum kernel so sánh feature hiện tại với prototype đã căn chỉnh:

\[
G_{ic}=|\langle\psi_{\omega}(f_i^t)
\mid\psi_{\omega}(P^t(\mu_c))\rangle|^2.
\]

Khi centering bật, feature và prototype trong cặp so sánh cùng dùng một tâm. Với mỗi feature, chọn \(\mathcal{N}_k(i)\), gồm \(k\) prototype cũ có fidelity cao nhất. Chuẩn hóa trọng số trên các prototype đã chọn rồi phạt các cặp còn fidelity vượt ngưỡng \(\varepsilon\):

\[
\alpha_{ic}=\frac{\exp(G_{ic}/\tau)}
{\sum_{j\in\mathcal{N}_k(i)}\exp(G_{ij}/\tau)},
\quad c\in\mathcal{N}_k(i),
\]

\[
\mathcal{L}_{\mathrm{qorth}}^t=
\frac{1}{B}\sum_i\sum_{c\in\mathcal{N}_k(i)}
\alpha_{ic}[G_{ic}-\varepsilon]_+.
\]

\(\tau\) điều khiển độ tập trung của softmax. Hinge bỏ qua cặp đã dưới ngưỡng, còn top-k giới hạn loss vào các lớp cũ có nguy cơ nhầm cao.

### 4.4. Loss tổng ở incremental task

\[
\beta(e)=\beta_{\max}\min(1,e/E_w),\qquad
\gamma(e)=\gamma_{\max}\min(1,e/E_w),
\]

\[
\mathcal{L}^{t}=\mathcal{L}_{\mathrm{cls}}^{t}
+\beta(e)\left(\mathcal{L}_{\mathrm{align}}^{t}
+\lambda_{\mathrm{qrel}}\mathcal{L}_{\mathrm{qrel}}^{t}\right)
+\gamma(e)\mathcal{L}_{\mathrm{qorth}}^{t},\qquad t>1.
\]

\(\mathcal{L}_{\mathrm{cls}}\) học nhãn lớp mới; \(\mathcal{L}_{\mathrm{align}}\) căn tọa độ; \(\mathcal{L}_{\mathrm{qrel}}\) giữ cấu trúc quan hệ batch; và \(\mathcal{L}_{\mathrm{qorth}}\) giảm giao thoa với prototype cũ gần nhất.

## 5. Thống kê lớp và pipeline dữ liệu

Ảnh cũ không được replay. Sau khi học mỗi task, trích xuất feature không augmentation của từng lớp mới rồi tính prototype và covariance:

\[
\mu_c=\frac{1}{n_c}\sum_{i:y_i=c}f_i,\qquad
\Sigma_c=\frac{1}{n_c-1}\sum_{i:y_i=c}
(f_i-\mu_c)(f_i-\mu_c)^\top+\epsilon I.
\]

Prototype cấp vector lớp cho qorth; covariance cùng prototype hỗ trợ sinh feature Gaussian để cập nhật classifier. Nếu bật hiệu chỉnh drift, dịch chuyển được ước lượng từ feature teacher/current trên ảnh task mới:

\[
w_{ic}=\exp\!\left(-\frac{\lVert f_i^{t-1}-\mu_c\rVert_2^2}{2\sigma^2}\right),\qquad
\delta_c=\frac{\sum_iw_{ic}(f_i^t-f_i^{t-1})}{\sum_iw_{ic}},\qquad
\mu_c\leftarrow\mu_c+\delta_c.
\]

Luồng dữ liệu và tín hiệu huấn luyện:

```mermaid
flowchart TD
    A[Ảnh và nhãn lớp mới] --> B[Backbone + adapter]
    B --> C[Feature hiện tại f^t]
    C --> D[Classifier và loss phân loại]
    C --> E[Quantum feature map]
    E --> F[Statevector psi(f)]
    F --> G[Fidelity theo cặp]
    G --> H[QRS ở base task]
    G --> I[Qrel ở incremental task]
    J[Teacher đóng băng] --> K[Feature tham chiếu f^(t-1)]
    K --> L[Residual quantum aligner]
    L --> M[Readout Pauli-X/Z: 2q giá trị]
    M --> N[Nhánh residual, khởi tạo identity]
    N --> O[Loss căn chỉnh]
    P[Prototype lớp cũ] --> N
    C --> Q[Quantum fidelity với prototype đã căn chỉnh]
    N --> Q
    Q --> R[Top-k + softmax + hinge qorth]
    D --> S[Tổng loss và cập nhật tham số]
    H --> S
    I --> S
    O --> S
    R --> S
    S --> T[Thống kê lớp mới: prototype/covariance]
    T --> U[Classifier cho task kế tiếp]
```

Theo từng task: (1) nạp ảnh mới và tạo batch; (2) trích xuất feature student, và nếu \(t>1\), feature teacher; (3) tạo statevector/fidelity cho các loss quantum; (4) tối ưu các nhánh đang hoạt động trong một graph; (5) tổng hợp prototype/covariance của lớp mới và cập nhật classifier; (6) đóng băng mạng hiện tại làm teacher cho task kế tiếp.

## 6. Suy luận

Đường dự đoán chỉ tính feature từ ảnh rồi chuyển qua classifier đã cập nhật:

\[
x\longrightarrow f(x)\longrightarrow\operatorname{classifier}
\longrightarrow\hat y.
\]

Không gọi statevector circuit, quantum feature map, fidelity kernel, teacher hay residual aligner ở suy luận. Vì vậy quantum branches không thêm phép tính vào dự đoán.

## 7. Cấu hình và vị trí triển khai

Cấu hình đầy đủ cho các nhánh lượng tử:

```json
{
  "aligner": "qhybrid",
  "lambda_qrel": 1.0,
  "kernel": "quantum",
  "orth": "qweighted",
  "rs_kernel": "quantum",
  "rs_margin": 0.3,
  "n_qubits": 8,
  "quantum_layers": 2,
  "quantum_centering": true
}
```

Base experiment cần có trọng số dương cho `beta` và `gamma`; `lambda_rs` cùng warm-up điều khiển QRS. Với fidelity \(F\in[0,1]\), margin negative phải nhỏ hơn 1 để còn tạo gradient đẩy các cặp khác lớp ra xa. Base config ImageNet-A đang đặt `rs_margin=1.5`; khi kết hợp với `rs_kernel="quantum"`, \([F_{ij}-1.5]_+=0\) cho mọi cặp âm, nên nhánh đẩy cặp âm bị tắt hoàn toàn. Cấu hình đầy đủ cần override `rs_margin` bằng giá trị trong \([0,1)\), rồi chọn giá trị phù hợp qua validation.

`E8_quantum_rs` trong `configs/experiments/ablation_full.json` bật đủ các nhánh quantum, nhưng hiện kế thừa `rs_margin=1.5` từ base ImageNet-A; vì vậy cần sửa margin như trên để QRS có cả lực hút và lực đẩy. `E9_deeper_circuit` tăng `quantum_layers` lên 3, nhưng cấu hình hiện tại không đặt `rs_kernel="quantum"`; cần thêm giá trị đó nếu muốn nhánh QRS cũng dùng mạch sâu hơn.

| Thành phần | Mã nguồn |
|---|---|
| Quantum state preparation và fidelity | `qrsiat/quantum/feature_map.py`, `qrsiat/quantum/simulator.py` |
| Pauli-X/Z readout và residual aligner | `qrsiat/quantum/aligner.py`, `qrsiat/quantum/simulator.py` |
| Kernel pairwise | `qrsiat/quantum/kernels.py` |
| Loss QRS, qrel, qorth và forward chung | `models/RSIAT_adapter.py`, `qrsiat/training/step_module.py`, `qrsiat/quantum/losses.py` |
| Prototype/covariance và drift | `models/base.py`, `qrsiat/stats/accumulators.py`, `qrsiat/stats/drift.py` |

Simulator dùng statevector thực với cổng RY/CNOT và mô phỏng trên máy cổ điển; mô tả này không hàm ý quantum advantage.
