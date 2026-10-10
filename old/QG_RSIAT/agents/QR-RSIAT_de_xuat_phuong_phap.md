# QR-RSIAT: Quantum-Relational Representation-Steered Incremental Adapter Tuning

*Đề xuất phương pháp kết hợp ý tưởng từ RSIAT (CVPR 2026) và QKD (CVPR 2026)*

---

## 1. Ý tưởng cốt lõi

- **Từ RSIAT:** một adapter dùng chung, định hình không gian đặc trưng ở base task ($\mathcal{L}_{\mathrm{RS}}$ + warm-up), căn chỉnh theo kiểu residual (đường skip đồng nhất) để không kìm plasticity, lưu prototype + covariance rồi sinh feature Gaussian để huấn luyện lại classifier.
- **Từ QKD:** mạch lượng tử tham số hóa tính độ tương quan bằng fidelity, và dùng độ tương quan đó làm **trọng số chọn lọc** (task/lớp nào liên quan thì được bảo vệ nhiều hơn).
- **Điểm khác:** cổng lượng tử **chỉ dùng khi huấn luyện** và bị bỏ khi suy luận. Suy luận chỉ gồm PTM đóng băng + adapter chung + classifier, nên tham số và latency không tăng, tránh nhược điểm của QKD (adapter mỗi task, latency 22.8 ms/ảnh).

## 2. Kiến trúc

### 2.1. Giai đoạn base task ($t=1$)

$$
\mathcal{L}^{(1)} = \mathcal{L}_{\cos}^{(1)} + \lambda_{\mathrm{RS}}(e)\,\mathcal{L}_{\mathrm{RS}}
$$

Giữ nguyên RSIAT. Tùy chọn: tính $\mathcal{L}_{\mathrm{RS}}$ bằng **kernel fidelity lượng tử** $K_{ij}$ thay cho cosine, để hình học ở base và incremental dùng cùng một feature map.

### 2.2. Giai đoạn incremental ($t>1$)

**(a) Bộ căn chỉnh lai cổ điển–lượng tử, khởi tạo đồng nhất (thay RAE)**

$$
P^{t}(f) = f + W_{\mathrm{up}}\,\mathrm{Readout}\!\left(\mathrm{PQC}\!\left(W_{\mathrm{down}} f\right)\right), \qquad W_{\mathrm{up}}^{(0)} = 0
$$

Vì có skip và $W_{\mathrm{up}}=0$ nên $P^{t}=I$ tại thời điểm đầu, giữ được tính chất "không có handbrake" của RSIAT. Mạch statevector thực dùng readout Pauli-$X/Z$ trên từng qubit, thu được $2q$ giá trị; Pauli-$Y$ bị lược vì kỳ vọng bằng 0 với trạng thái thực:

$$
\mathrm{Readout}(\psi) = \Big[\langle X_j\rangle_{j=1}^{q},\;\langle Z_j\rangle_{j=1}^{q}\Big]
$$

Mạch nông, $l_q \le 2$–$3$, $q \approx 8$–$12$.

**(b) Alignment quan hệ bằng kernel lượng tử (đóng góp chính)**

Với batch $\mathcal{B}$, mã hóa chuẩn hóa $z \mapsto \vert\psi(z)\rangle$ bằng cùng một feature map, rồi tính:

$$
K^{t-1}_{ij} = \left\lvert \langle \psi(f_i^{t-1}) \mid \psi(f_j^{t-1}) \rangle \right\rvert^{2},
\qquad
K^{t}_{ij} = \left\lvert \langle \psi(f_i^{t}) \mid \psi(f_j^{t}) \rangle \right\rvert^{2}
$$

$$
\mathcal{L}_{\mathrm{qrel}} = \left\lVert K^{t} - \mathrm{sg}\!\left(K^{t-1}\right) \right\rVert_F^{2}
$$

Khớp **cấu trúc quan hệ** thay vì khớp tuyệt đối nên bất biến với phép xoay toàn cục (drift), ít cản adapter hơn. Cùng tinh thần R-DFCIL / Quantum Relational KD. Có thể dùng L2 tuyệt đối qua $P^{t}$ với trọng số nhỏ làm thành phần phụ:

$$
\mathcal{L}_{\mathrm{align}} = \frac{1}{\lvert\mathcal{B}\rvert}\sum_{x\in\mathcal{B}} \left\lVert f^{t} - P^{t}\!\left(f^{t-1}\right) \right\rVert_2^{2}
$$

**(c) Trực giao có trọng số chọn lọc, lấy ý tưởng từ QGTM**

Với feature mới $\hat u_i$ và prototype cũ $\hat p_c$, prototype cũ được ánh xạ qua $P^{t}$ trước khi tính fidelity. Khi centering bật, feature và prototype trong từng phép so cặp dùng chung tâm. Drift ước lượng ở cuối task không được đưa ngược vào loss của chính task đó; bù drift cần được đánh giá riêng bằng thông tin có sẵn trước hoặc cập nhật online:

$$
F_{ic} = \left\lvert \langle \psi(\hat u_i) \mid \psi(\hat p_c) \rangle \right\rvert^{2},
\qquad
\alpha_{ic} = \frac{\exp(F_{ic}/\tau)}{\sum_{c'} \exp(F_{ic'}/\tau)}, \quad \tau \text{ nhỏ}
$$

$$
\mathcal{L}_{\mathrm{orth}}^{q} = \frac{1}{\lvert\mathcal{B}\rvert}\sum_{i\in\mathcal{B}} \sum_{c\in \mathrm{top}\text{-}k_i} \alpha_{ic}\,\max\!\left(0,\; F_{ic} - \varepsilon\right)
$$

Tức là chỉ phạt các lớp cũ **dễ nhầm nhất** với mẫu hiện tại ($\mathrm{top}\text{-}k_i$ là $k$ lớp cũ có $F_{ic}$ lớn nhất với mẫu $i$), thay vì phạt đều như $\mathcal{L}_{\mathrm{orth}}$ của RSIAT (vốn làm $A_B$ giảm ở Bảng 3). Prototype cũ được đưa qua cùng aligner $P^t$; khi centering bật, feature và prototype dùng chung tâm trong từng phép so cặp. Hinge cho phép "đã đủ trực giao thì thôi", ít hy sinh plasticity hơn.

**(d) Classifier:** giữ nguyên SSIAT/RSIAT, tức prototype + covariance → sinh feature Gaussian → huấn luyện lại head.

### 2.3. Loss tổng

$$
\mathcal{L}^{(t)} = \mathcal{L}_{\cos}^{(t)} + \beta(e)\left(\mathcal{L}_{\mathrm{qrel}} + \mathcal{L}_{\mathrm{align}}\right) + \gamma(e)\,\mathcal{L}_{\mathrm{orth}}^{q}, \qquad t>1
$$

$$
\mathcal{L}^{(1)} = \mathcal{L}_{\cos}^{(1)} + \lambda_{\mathrm{RS}}(e)\,\mathcal{L}_{\mathrm{RS}}
$$

với $\beta(e)$ và $\gamma(e)$ warm-up tuyến tính như $\lambda_{\mathrm{RS}}$:

$$
\beta(e) = \beta\cdot\min\!\left(1, \frac{e}{E_w}\right), \qquad \gamma(e) = \gamma\cdot\min\!\left(1, \frac{e}{E_w}\right)
$$

## 3. Nguồn gốc từng thành phần

| Thành phần | Nguồn | Thay đổi |
|---|---|---|
| Shared adapter, $\mathcal{L}_{\cos}$, $\mathcal{L}_{\mathrm{RS}}$ + warm-up | RSIAT | giữ nguyên |
| Skip đồng nhất, init bằng 0 | RSIAT | áp dụng cho bộ căn chỉnh lai |
| Prototype + covariance, bù drift | SSIAT/RSIAT | giữ nguyên |
| Mã hóa lượng tử + fidelity | QKD | dùng nông, PQK, chỉ khi train |
| Trọng số liên quan sample→task | QKD | chuyển thành trọng số hard-negative cho $\mathcal{L}_{\mathrm{orth}}$ |
| Sparsity $\mathcal{L}_s$ | QKD | **bỏ** (xem mục 4) |

## 4. Những lỗi cần tránh (rút ra từ QKD)

- **$\mathcal{L}_s = \lVert\alpha\rVert_1$ sau softmax luôn bằng 1**, vì $\alpha_i>0$ và $\sum_i \alpha_i = 1$ nên gradient bằng 0. Nếu cần ép thưa thì dùng top-$k$ hoặc entropy của $\alpha$:
  $$H(\alpha) = -\sum_i \alpha_i \log \alpha_i$$
- Với $p\in[0,1]$, softmax với $\tau=1$ cho trọng số gần đều. Dùng $\tau$ nhỏ hoặc log-fidelity $\log(F+\epsilon)$.
- Centering feature trước khi mã hóa, $\tilde f = f - \bar f$, vì feature ViT dị hướng nên cos/fidelity giữa các mẫu thường đã cao.
- Kiểm tra gradient norm $\lVert \nabla_\theta F \rVert$ theo độ sâu mạch để phát hiện barren plateau/concentration sớm.

## 5. Thực nghiệm đề xuất

**Nền so sánh:** RSIAT gốc (có RAE), bỏ projector + L2, bỏ projector + cosine, QKD, MOS, SSIAT.

**Ablation chính:**

1. Bộ căn chỉnh: RAE vs lai-lượng-tử (cùng số tham số).
2. Kernel trong $\mathcal{L}_{\mathrm{qrel}}$: fidelity lượng tử vs cosine vs RBF vs MLP-kernel. Đây là ablation quyết định để bảo vệ lý do dùng lượng tử.
3. $\mathcal{L}_{\mathrm{orth}}$: đều vs có trọng số $\alpha$/top-$k$ + hinge.
4. Cho/không cho skip đồng nhất (lặp lại thí nghiệm `wo_res`).
5. $q\in\{4,8,12\}$, $l_q\in\{1,2,3\}$.

**Chỉ số:** $\bar{\mathcal{A}}$, $\mathcal{A}_B$, độ trôi prototype, Recall@1, thời gian train, thời gian suy luận.

**Dữ liệu:** IN-A, IN-R (nơi RSIAT mạnh nhất), CUB, VTAB, thêm chuỗi dài B0 Inc10 và large-base như Hình 3–4 của RSIAT. Chạy $\ge 3$ seed, báo cáo mean $\pm$ std, vì chênh lệch dưới $\sim 0.5$ điểm của cả hai bài gốc chưa chắc phân biệt được với nhiễu.

## 6. Rủi ro và phương án dự phòng

- **Lượng tử không hơn cổ điển ở ablation 2:** vẫn có bài báo nếu trình bày đóng góp là "relational alignment + selective orthogonality, kernel-agnostic", lượng tử chỉ là một lựa chọn kernel. Đừng khẳng định "quantum advantage".
- **Chi phí mô phỏng:** với $q\le 12$ và $\lvert\mathcal{B}\rvert=32$ thì statevector rất rẻ ($2^{12}=4096$ chiều). Chỉ chạy lúc train.
- **$\mathcal{L}_{\mathrm{qrel}}$ quá yếu để chống drift:** tăng trọng số $\mathcal{L}_{\mathrm{align}}$ tuyệt đối và bật dần theo warm-up.
